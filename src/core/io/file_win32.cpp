// Win32 backend of core/io/file.h. A ReadOnlyFile owns one synchronous handle: Buffered mode is an
// ordinary cached handle, Direct mode is opened with FILE_FLAG_NO_BUFFERING (the O_DIRECT
// equivalent) and reports the volume's sector alignment. Positional reads pass the file offset in
// an OVERLAPPED, so they never depend on (or race over) a shared file pointer. Sharing is as
// permissive as POSIX open: a reader never blocks another reader, a writer, or a rename/delete.
#include "core/io/file.h"

#ifndef NOMINMAX
#    define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>

#include <algorithm>
#include <bit>
#include <cerrno>
#include <limits>
#include <malloc.h>
#include <memory>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>

namespace ninfer::io {
namespace {

constexpr auto kMaximumOffset = static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max());

std::string display(const std::filesystem::path& path) {
    const auto text = path.u8string();
    return {reinterpret_cast<const char*>(text.data()), text.size()};
}

[[noreturn]] void fail_system(const std::filesystem::path& path, const char* operation,
                              DWORD error = ::GetLastError()) {
    throw std::system_error(static_cast<int>(error), std::system_category(),
                            display(path) + ": " + operation);
}

[[noreturn]] void fail_generic(const std::filesystem::path& path, const char* operation,
                               int error) {
    throw std::system_error(error, std::generic_category(), display(path) + ": " + operation);
}

HANDLE handle_of(FileDescriptor descriptor) noexcept {
    return reinterpret_cast<HANDLE>(descriptor);
}

void close_descriptor(FileDescriptor descriptor) noexcept {
    if (descriptor >= 0) ::CloseHandle(handle_of(descriptor));
}

void check_position(std::uint64_t value) {
    if (value > kMaximumOffset) throw std::invalid_argument("file offset exceeds off_t");
}

// One synchronous positional ReadFile. Returns the byte count (zero at or beyond EOF); throws
// on any other failure.
DWORD read_at(const std::filesystem::path& path, HANDLE handle, std::uint64_t offset, void* data,
              DWORD bytes, const char* operation) {
    OVERLAPPED position{};
    position.Offset     = static_cast<DWORD>(offset & 0xffffffffULL);
    position.OffsetHigh = static_cast<DWORD>(offset >> 32U);
    DWORD read          = 0;
    if (!::ReadFile(handle, data, bytes, &read, &position)) {
        const auto error = ::GetLastError();
        if (error == ERROR_HANDLE_EOF) return 0;
        fail_system(path, operation, error);
    }
    return read;
}

struct AlignedFree {
    void operator()(void* data) const noexcept { _aligned_free(data); }
};

} // namespace

ReadOnlyFile::ReadOnlyFile(const std::filesystem::path& path, ReadMode mode) : path_(path) {
    if (mode != ReadMode::Buffered && mode != ReadMode::Direct)
        throw std::invalid_argument("invalid file read mode");
    const DWORD flags =
        FILE_ATTRIBUTE_NORMAL | (mode == ReadMode::Direct ? FILE_FLAG_NO_BUFFERING : 0);
    const HANDLE handle =
        ::CreateFileW(path.c_str(), GENERIC_READ,
                      FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr,
                      OPEN_EXISTING, flags, nullptr);
    if (handle == INVALID_HANDLE_VALUE) fail_system(path, "open");
    view_.descriptor = reinterpret_cast<FileDescriptor>(handle);
    view_.mode       = mode;
    try {
        if (::GetFileType(handle) != FILE_TYPE_DISK)
            fail_generic(path, "expected regular file", EINVAL);
        BY_HANDLE_FILE_INFORMATION information{};
        if (!::GetFileInformationByHandle(handle, &information)) fail_system(path, "fstat");
        if (information.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)
            fail_generic(path, "expected regular file", EINVAL);
        LARGE_INTEGER size{};
        if (!::GetFileSizeEx(handle, &size)) fail_system(path, "fstat");
        if (size.QuadPart < 0) fail_generic(path, "expected regular file", EINVAL);
        view_.bytes = static_cast<std::uint64_t>(size.QuadPart);
        if (mode == ReadMode::Direct) {
            // Unbuffered I/O needs offsets and sizes in whole volume sectors and buffers aligned
            // to the physical sector; one power-of-two unit covering both serves as both
            // alignments, as on Linux without STATX_DIOALIGN (4 KiB on typical NVMe volumes).
            FILE_STORAGE_INFO storage{};
            const bool reported = ::GetFileInformationByHandleEx(handle, FileStorageInfo, &storage,
                                                                 sizeof(storage)) != 0 &&
                                  storage.LogicalBytesPerSector != 0;
            const std::size_t unit =
                reported ? std::max<std::size_t>(storage.LogicalBytesPerSector,
                                                 storage.PhysicalBytesPerSectorForPerformance)
                         : 4096;
            view_.memory_alignment = unit;
            view_.offset_alignment = unit;
            if (!std::has_single_bit(view_.memory_alignment) ||
                !std::has_single_bit(view_.offset_alignment))
                fail_generic(path, "unsupported direct I/O alignment", EOPNOTSUPP);
            if (!reported && view_.bytes) {
                // Without a reported sector size, prove the engine's 4 KiB read unit works.
                std::unique_ptr<void, AlignedFree> probe(_aligned_malloc(4096, 4096));
                if (!probe) throw std::bad_alloc();
                (void)read_direct(0, {static_cast<std::byte*>(probe.get()), 4096});
            }
        }
    } catch (...) {
        close_descriptor(view_.descriptor);
        view_.descriptor = -1;
        throw;
    }
}

ReadOnlyFile::~ReadOnlyFile() { close_descriptor(view_.descriptor); }

ReadOnlyFile::ReadOnlyFile(ReadOnlyFile&& other) noexcept
    : path_(std::move(other.path_)), view_(std::exchange(other.view_, {})) {}

ReadOnlyFile& ReadOnlyFile::operator=(ReadOnlyFile&& other) noexcept {
    if (this != &other) {
        close_descriptor(view_.descriptor);
        path_ = std::move(other.path_);
        view_ = std::exchange(other.view_, {});
    }
    return *this;
}

void ReadOnlyFile::read_exact(std::uint64_t offset, std::span<std::byte> destination) const {
    if (view_.mode != ReadMode::Buffered || view_.descriptor < 0)
        throw std::invalid_argument("read_exact requires a buffered file");
    if (offset > view_.bytes || destination.size() > view_.bytes - offset)
        fail_generic(path_, "read exceeds file length", ENODATA);
    while (!destination.empty()) {
        const auto count = std::min<std::size_t>(destination.size(), 64ULL * 1024 * 1024);
        check_position(offset);
        const auto n = read_at(path_, handle_of(view_.descriptor), offset, destination.data(),
                               static_cast<DWORD>(count), "pread");
        if (!n) fail_generic(path_, "unexpected EOF", ENODATA);
        offset += n;
        destination = destination.subspan(n);
    }
}

void validate_direct_read(FileView file, std::uint64_t offset, std::span<std::byte> destination) {
    if (file.descriptor < 0 || file.mode != ReadMode::Direct ||
        !std::has_single_bit(file.memory_alignment) ||
        !std::has_single_bit(file.offset_alignment) || offset % file.offset_alignment ||
        destination.size() % file.offset_alignment ||
        (!destination.empty() &&
         (!destination.data() ||
          reinterpret_cast<std::uintptr_t>(destination.data()) % file.memory_alignment)) ||
        destination.size() > static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max()))
        throw std::invalid_argument("invalid direct read range or alignment");
    check_position(offset);
}

std::size_t ReadOnlyFile::read_direct(std::uint64_t offset,
                                      std::span<std::byte> destination) const {
    validate_direct_read(view_, offset, destination);
    if (destination.empty()) return 0;
    // ReadFile takes a DWORD count: split the request into sector-aligned 1 GiB reads, stopping
    // at the first short one (EOF), so the result matches one POSIX direct pread.
    constexpr std::size_t kMaximumRead = 1ULL << 30;
    std::size_t total                  = 0;
    while (total < destination.size()) {
        const auto amount = static_cast<DWORD>(std::min(kMaximumRead, destination.size() - total));
        check_position(offset + total);
        const auto n = read_at(path_, handle_of(view_.descriptor), offset + total,
                               destination.data() + total, amount, "direct pread");
        total += n;
        if (n != amount) break;
    }
    return total;
}

DirectReadWindow direct_read_window(FileView file, std::uint64_t offset, std::size_t bytes) {
    if (file.descriptor < 0 || file.mode != ReadMode::Direct || !bytes ||
        !std::has_single_bit(file.offset_alignment) || offset > file.bytes ||
        bytes > file.bytes - offset)
        throw std::invalid_argument("invalid direct read window");
    const auto alignment = file.offset_alignment;
    const auto prefix    = offset % alignment;
    if (bytes > std::numeric_limits<std::size_t>::max() - prefix - (alignment - 1))
        throw std::overflow_error("direct read window size overflow");
    const auto required = bytes + prefix;
    const auto count    = (required + alignment - 1) / alignment * alignment;
    check_position(offset - prefix);
    return {offset - prefix, count, prefix, required};
}

} // namespace ninfer::io
