#include "logger.hpp"
#include <cstdio>  // For vprintf

// Log level to string mapping
const char* Logger::levelToString(int level) {
    switch (level) {
        case LOG_LEVEL_DEBUG:   return "DEBUG";
        case LOG_LEVEL_INFO:    return "INFO";
        case LOG_LEVEL_WARNING: return "WARNING";
        case LOG_LEVEL_ERROR:   return "ERROR";
        default:                return "UNKNOWN";
    }
}

// Log function implementation
void Logger::log(int level, const char* format, ...) {
    const char* levelStr = levelToString(level);

    // Print log level
    std::cout << "[" << levelStr << "] ";

    // Handle variadic arguments
    va_list args;
    va_start(args, format);
    vprintf(format, args);
    va_end(args);

    std::cout << std::endl;
}
