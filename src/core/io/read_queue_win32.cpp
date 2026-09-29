// Win32 backend of core/io/read_queue.h: overlapped unbuffered ReadFile on an I/O completion port
// in place of io_uring. Each queue reopens its files (ReOpenFile) as overlapped unbuffered
// handles associated with its own port, since a handle can belong to only one port and the
// borrowed FileView handles are synchronous. Accepted reads are issued by submit(). Packets the
// port delivers are held until poll() returns them, so wait() can block on the port without
// consuming completions; wake() posts one coalesced packet with its own key.
#include "core/io/read_queue.h"

#ifndef NOMINMAX
#    define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <deque>
#include <exception>
#include <limits>
#include <stdexcept>
#include <system_error>
#include <vector>

namespace ninfer::io {
namespace {

constexpr ULONG_PTR kReadKey = 0;
constexpr ULONG_PTR kWakeKey = 1;

[[noreturn]] void fail(const char* operation, DWORD error = ::GetLastError()) {
    throw std::system_error(static_cast<int>(error), std::system_category(), operation);
}

// ReadCompletion::error is a positive errno, as io_uring reports it.
int errno_from(DWORD error) noexcept {
    switch (error) {
    case ERROR_SUCCESS: return 0;
    case ERROR_INVALID_PARAMETER:
    case ERROR_INVALID_USER_BUFFER:
    case ERROR_OFFSET_ALIGNMENT_VIOLATION: return EINVAL;
    case ERROR_NOT_ENOUGH_MEMORY:
    case ERROR_OUTOFMEMORY:
    case ERROR_NO_SYSTEM_RESOURCES:
    case ERROR_WORKING_SET_QUOTA:
    case ERROR_NOT_ENOUGH_QUOTA: return ENOMEM;
    case ERROR_ACCESS_DENIED: return EACCES;
    case ERROR_INVALID_HANDLE: return EBADF;
    case ERROR_OPERATION_ABORTED: return ECANCELED;
    default: return EIO;
    }
}

} // namespace

struct DirectReadQueue::Impl {
    struct Slot {
        DirectRead request{};
        bool busy = false;
    };

    struct Finished {
        std::uint32_t index;
        std::uint32_t bytes;
        int error;
    };

    HANDLE port            = nullptr;
    bool closed            = false;
    bool signalled         = false; // A wake packet was consumed and not yet reported by wait().
    std::uint32_t capacity = 0;
    std::size_t pending    = 0;
    std::atomic<bool> wake_posted{false};
    std::vector<FileView> files;
    std::vector<HANDLE> handles;
    std::vector<MemorySpan> buffers;
    std::vector<Slot> slots;
    std::vector<OVERLAPPED> operations; // One per buffer; stable for the queue's lifetime.
    std::vector<std::uint32_t> prepared;
    std::deque<Finished> finished;

    ~Impl() {
        for (const auto handle : handles) ::CloseHandle(handle);
        if (port) ::CloseHandle(port);
    }

    void finish(std::uint32_t index, DWORD bytes, DWORD error) {
        // Reading at or past EOF is a successful zero-byte (or short) read, as with pread.
        if (error == ERROR_HANDLE_EOF) error = ERROR_SUCCESS;
        finished.push_back({index, error ? 0U : static_cast<std::uint32_t>(bytes), errno_from(error)});
    }

    // Moves delivered packets into `finished`; waits up to `timeout` for the first one.
    // Returns whether any packet (read or wake) arrived.
    bool harvest(DWORD timeout) {
        std::array<OVERLAPPED_ENTRY, 64> entries{};
        ULONG count = 0;
        if (!::GetQueuedCompletionStatusEx(port, entries.data(), static_cast<ULONG>(entries.size()),
                                           &count, timeout, FALSE)) {
            const auto error = ::GetLastError();
            if (error == WAIT_TIMEOUT) return false;
            fail("wait for direct reads", error);
        }
        for (ULONG i = 0; i < count; ++i) {
            const auto& entry = entries[i];
            if (entry.lpCompletionKey == kWakeKey) {
                wake_posted.exchange(false, std::memory_order_acq_rel);
                signalled = true;
                continue;
            }
            const auto* operation = entry.lpOverlapped;
            if (entry.lpCompletionKey != kReadKey || operation < operations.data() ||
                operation >= operations.data() + operations.size())
                throw std::logic_error("unmatched direct read completion");
            const auto index = static_cast<std::uint32_t>(operation - operations.data());
            if (!slots[index].busy) throw std::logic_error("unmatched direct read completion");
            DWORD bytes = 0;
            DWORD error = ERROR_SUCCESS;
            if (!::GetOverlappedResult(handles[slots[index].request.file_index],
                                       &operations[index], &bytes, FALSE))
                error = ::GetLastError();
            finish(index, bytes, error);
        }
        return count != 0;
    }
};

DirectReadQueue::DirectReadQueue(std::span<const FileView> files,
                                 std::span<const MemorySpan> buffers, std::uint32_t capacity)
    : impl_(std::make_unique<Impl>()) {
    if (!capacity || capacity > buffers.size() || files.empty() ||
        files.size() > std::numeric_limits<int>::max() || buffers.size() > 65535)
        throw std::invalid_argument("invalid direct read queue capacity");
    auto& p    = *impl_;
    p.capacity = capacity;
    p.files.assign(files.begin(), files.end());
    p.buffers.assign(buffers.begin(), buffers.end());
    p.slots.resize(buffers.size());
    p.operations.resize(buffers.size());
    for (const auto file : files) {
        if (file.mode != ReadMode::Direct || file.descriptor < 0)
            throw std::invalid_argument("read queue requires direct files");
    }
    for (std::size_t i = 0; i < buffers.size(); ++i) {
        const auto buffer = buffers[i];
        validate_memory(buffer);
        if (buffer.device != Device::cpu() || !buffer.bytes ||
            buffer.bytes > std::numeric_limits<unsigned>::max())
            throw std::invalid_argument("read queue requires nonempty Host buffers");
        for (const auto file : files)
            validate_direct_read(file, 0, {static_cast<std::byte*>(buffer.data), buffer.bytes});
        const auto begin = reinterpret_cast<std::uintptr_t>(buffer.data);
        for (std::size_t j = 0; j < i; ++j) {
            const auto other = reinterpret_cast<std::uintptr_t>(buffers[j].data);
            if (begin < other + buffers[j].bytes && other < begin + buffer.bytes)
                throw std::invalid_argument("read queue buffers overlap");
        }
    }
    p.port = ::CreateIoCompletionPort(INVALID_HANDLE_VALUE, nullptr, 0, 1);
    if (!p.port) fail("initialize direct read queue");
    p.handles.reserve(files.size());
    for (const auto file : files) {
        const HANDLE handle = ::ReOpenFile(reinterpret_cast<HANDLE>(file.descriptor), GENERIC_READ,
                                           FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                                           FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED);
        if (handle == INVALID_HANDLE_VALUE) fail("register read files");
        p.handles.push_back(handle);
        if (::CreateIoCompletionPort(handle, p.port, kReadKey, 0) != p.port)
            fail("register read files");
    }
}

DirectReadQueue::~DirectReadQueue() {
    try {
        close();
        drain();
    } catch (...) {
        // Releasing borrowed buffers with unknown kernel writers would violate ownership.
        std::terminate();
    }
}

ReadAdmission DirectReadQueue::try_enqueue(DirectRead request) {
    auto& p = *impl_;
    if (p.closed) return ReadAdmission::Closed;
    if (p.pending == p.capacity) return ReadAdmission::Full;
    if (request.file_index >= p.files.size() || request.buffer_index >= p.buffers.size() ||
        !request.bytes)
        throw std::invalid_argument("invalid direct read request");
    auto& slot        = p.slots[request.buffer_index];
    const auto buffer = p.buffers[request.buffer_index];
    if (slot.busy || request.bytes > buffer.bytes)
        throw std::invalid_argument("direct read buffer is busy or too small");
    validate_direct_read(p.files[request.file_index], request.file_offset,
                         {static_cast<std::byte*>(buffer.data), request.bytes});
    slot.request = request;
    slot.busy    = true;
    p.prepared.push_back(request.buffer_index);
    ++p.pending;
    return ReadAdmission::Accepted;
}

void DirectReadQueue::submit() {
    auto& p = *impl_;
    for (const auto index : p.prepared) {
        const auto& request = p.slots[index].request;
        auto& operation     = p.operations[index];
        operation            = {};
        operation.Offset     = static_cast<DWORD>(request.file_offset & 0xffffffffULL);
        operation.OffsetHigh = static_cast<DWORD>(request.file_offset >> 32U);
        // Without FILE_SKIP_COMPLETION_PORT_ON_SUCCESS the port receives a packet whether the
        // read completes inline or later; only an immediate failure produces none.
        if (!::ReadFile(p.handles[request.file_index], p.buffers[index].data, request.bytes,
                        nullptr, &operation)) {
            const auto error = ::GetLastError();
            if (error != ERROR_IO_PENDING) p.finish(index, 0, error);
        }
    }
    p.prepared.clear();
}

std::size_t DirectReadQueue::poll(std::span<ReadCompletion> output) {
    auto& p           = *impl_;
    std::size_t count = 0;
    while (count < output.size()) {
        if (p.finished.empty() && !p.harvest(0)) break;
        if (p.finished.empty()) continue;
        const auto done = p.finished.front();
        p.finished.pop_front();
        auto& slot      = p.slots[done.index];
        output[count++] = {slot.request.tag, done.index, done.bytes, done.error};
        slot.busy       = false;
        --p.pending;
    }
    return count;
}

ReadWake DirectReadQueue::wait() {
    auto& p = *impl_;
    for (;;) {
        if (p.signalled) {
            p.signalled = false;
            return ReadWake::Signalled;
        }
        if (!p.finished.empty()) return ReadWake::Completion;
        (void)p.harvest(INFINITE);
    }
}

void DirectReadQueue::wake() {
    auto& p = *impl_;
    if (p.wake_posted.exchange(true, std::memory_order_acq_rel)) return;
    if (!::PostQueuedCompletionStatus(p.port, 0, kWakeKey, nullptr)) {
        const auto error = ::GetLastError();
        p.wake_posted.store(false, std::memory_order_release);
        fail("wake direct read queue", error);
    }
}

void DirectReadQueue::close() {
    impl_->closed = true;
    submit();
}

void DirectReadQueue::drain() {
    submit();
    std::array<ReadCompletion, 128> completions;
    while (impl_->pending)
        if (!poll(completions)) (void)wait();
}

std::size_t DirectReadQueue::outstanding() const noexcept { return impl_->pending; }

} // namespace ninfer::io
