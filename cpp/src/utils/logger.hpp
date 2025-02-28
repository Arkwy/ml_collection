#ifndef LOGGER_H
#define LOGGER_H

#include <cstdarg>
#include <iostream>

// Define log levels as constants
#define LOG_LEVEL_DEBUG 0
#define LOG_LEVEL_INFO 1
#define LOG_LEVEL_WARNING 2
#define LOG_LEVEL_ERROR 3

// Set the active log level at compile-time (Change this value)
#ifndef LOG_LEVEL
#define LOG_LEVEL LOG_LEVEL_DEBUG  // Default log level
#endif

// Logging macro
#define LOG(level, message, ...)                        \
	if constexpr (level >= LOG_LEVEL) {                 \
        Logger::log(level, message, ##__VA_ARGS__); \
	}

// Logger class declaration
class Logger {
  public:
	static void log(int level, const char* format, ...);

  private:
	static const char* levelToString(int level);
};

#endif	// LOGGER_H
