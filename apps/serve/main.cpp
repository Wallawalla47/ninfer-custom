#include "ninfer_build_id.h"

#include "product/logging/logging.h"
#include "product/logging/engine_diagnostics.h"
#include "product/logging/startup_log.h"
#include "serve/operational_log.h"
#include "serve/generation_service.h"
#include "serve/http_server.h"
#include "serve/serve_options.h"

#include <spdlog/logger.h>

#include <atomic>
#include <chrono>
#include <csignal>
#include <exception>
#include <filesystem>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>

#ifdef _WIN32
#    include <windows.h>
#endif

namespace {

std::atomic<ninfer::serve::HttpServer*> g_server{nullptr};

void handle_signal(int) {
    ninfer::serve::HttpServer* server = g_server.load();
    if (server != nullptr) { server->stop(); }
}

#ifdef _WIN32
// Ctrl+C reaches handle_signal through the C runtime; the other console events end the process
// without it. Ctrl+Break stops the server like Ctrl+C. Closing the console window terminates the
// process as soon as this handler returns, so it stops the server and then blocks: main unwinds,
// the Engine saves the prefix cache (--prefix-cache-file), and the process exits normally.
// Windows terminates it anyway about 5 seconds after the close; an unfinished save leaves the
// previous cache file in place.
BOOL WINAPI handle_console_event(DWORD event) {
    switch (event) {
    case CTRL_BREAK_EVENT:
        handle_signal(SIGINT);
        return TRUE;
    case CTRL_CLOSE_EVENT:
        handle_signal(SIGINT);
        Sleep(INFINITE);
        return TRUE;
    default:
        return FALSE;
    }
}
#endif

// Identity of this exact binary for the persisted prefix cache: the build id alone repeats for
// every uncommitted build, so the executable's size and modification time are included and any
// rebuild invalidates a saved cache whose bytes it may compute differently.
std::string binary_identity(const char* argv0) {
    std::string out;
#ifdef NINFER_BUILD_ID
    out = NINFER_BUILD_ID;
#endif
    std::error_code error;
#ifdef _WIN32
    // argv[0] is whatever the shell typed, which for a PATH launch is not a path to this file.
    (void)argv0;
    std::wstring module(MAX_PATH, L'\0');
    DWORD length = 0;
    while ((length = GetModuleFileNameW(nullptr, module.data(),
                                        static_cast<DWORD>(module.size()))) == module.size()) {
        module.resize(module.size() * 2);
    }
    module.resize(length);
    const std::filesystem::path self(module);
#else
    std::filesystem::path self = std::filesystem::read_symlink("/proc/self/exe", error);
    if (error) { self = std::filesystem::absolute(argv0, error); }
#endif
    const auto size = std::filesystem::file_size(self, error);
    if (!error) { out += ";size=" + std::to_string(size); }
    const auto time = std::filesystem::last_write_time(self, error);
    if (!error) { out += ";mtime=" + std::to_string(time.time_since_epoch().count()); }
    return out;
}

} // namespace

int main(int argc, char** argv) {
    ninfer::serve::ServeOptions options;
    try {
        options = ninfer::serve::parse_serve_options(argc, argv);
    } catch (const std::invalid_argument& exception) {
        std::cerr << "ninfer-serve: " << exception.what() << '\n';
        std::cerr << ninfer::serve::serve_usage_text(argv[0]);
        return 1;
    } catch (const std::exception& exception) {
        std::cerr << "ninfer-serve: " << exception.what() << '\n';
        return 1;
    }
    if (options.help_requested) {
        std::cout << ninfer::serve::serve_usage_text(argv[0]);
        return 0;
    }
    if (!options.context_cache.hybrid.persistent_file.empty()) {
        options.context_cache.hybrid.persistent_identity = binary_identity(argv[0]);
    }
    // Token/level colouring is opt-in (--log-colours on); the default keeps the log plain.
    ninfer::serve::set_operational_log_colours(options.log_colours);

    ninfer::product::LoggingOptions logging_options;
    logging_options.logger_name  = "ninfer-serve";
    logging_options.level        = options.log_level;
    logging_options.presentation = ninfer::product::LogPresentation::Service;
    logging_options.color        = options.log_colours
                                       ? ninfer::product::LogColorMode::Always
                                       : ninfer::product::LogColorMode::Never;
    ninfer::product::LoggingRuntime logging(logging_options);
    const std::shared_ptr<spdlog::logger> logger = logging.logger();
    ninfer::product::StartupLogRenderer startup_log(logging);
    ninfer::serve::OperationalLog operational_log(logger);
    bool serving = false;

    try {
        ninfer::serve::HttpServer server(options, logger, logging.terminal_panel());
        if (!server.bind()) {
            operational_log.bind_failure(options.host, options.port);
            return 1;
        }

#ifdef NINFER_BUILD_ID
        logger->info("build {}", NINFER_BUILD_ID);
#endif
        ninfer::serve::GenerationService service(
            options, startup_log.observer(), ninfer::product::engine_diagnostic_observer(logger));
        startup_log.engine_ready(service.load_summary());
        operational_log.engine_capacity(service);

        using Clock                            = std::chrono::steady_clock;
        const Clock::time_point warmup_started = Clock::now();
        operational_log.warmup_started();
        try {
            service.warmup();
        } catch (const std::exception& exception) {
            const double seconds =
                std::chrono::duration<double>(Clock::now() - warmup_started).count();
            operational_log.warmup_failure(seconds, exception.what());
            return 1;
        }
        operational_log.warmup_complete(
            std::chrono::duration<double>(Clock::now() - warmup_started).count());
        server.attach(service);

        g_server.store(&server);
        std::signal(SIGINT, handle_signal);
        std::signal(SIGTERM, handle_signal);
#ifdef _WIN32
        SetConsoleCtrlHandler(handle_console_event, TRUE);
#endif

        serving = true;
        operational_log.server_ready(options.host, options.port, server.public_model_id(),
                                     !options.api_key.empty());

        const bool ok = server.listen();
        g_server.store(nullptr);
        if (!ok) {
            operational_log.listen_failure(options.host, options.port);
            return 1;
        }
        operational_log.server_stopped();
        return 0;
    } catch (const std::exception& exception) {
        g_server.store(nullptr);
        operational_log.server_failure(serving, exception.what());
        return 1;
    }
}
