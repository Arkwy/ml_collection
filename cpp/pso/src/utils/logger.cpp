#include "logger.hpp"

#include <iostream>
#include <cstdio>

const char* Logger::levelToString(int level) {
    switch (level) {
        case LOG_LEVEL_DEBUG:
            return "DEBUG";
        case LOG_LEVEL_INFO:
            return "INFO";
        case LOG_LEVEL_WARNING:
            return "WARNING";
        case LOG_LEVEL_ERROR:
            return "ERROR";
        default:
            return "UNKNOWN";
    }
}

void Logger::log(int level, const char* format, ...) {
    const char* levelStr = levelToString(level);

    std::cout << "[" << levelStr << "] ";

    // Handle variadic arguments
    va_list args;
    va_start(args, format);
    vprintf(format, args);
    va_end(args);

    std::cout << std::endl;
}
