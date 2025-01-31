// Logger.h
#ifndef LOGGER_H
#define LOGGER_H

#include <iostream>
#include <string>
#include <mutex>

// Log levels
enum class LogLevel {
    DEBUG,
    INFO,
    WARN,
    ERROR,
    FATAL
};

class Logger {
private:
    LogLevel currentLevel;
    std::ostream* outputStream;
    std::mutex logMutex;

    // Private constructor (Singleton)
    Logger();

public:
    // Delete copy constructor and assignment operator
    Logger(const Logger&) = delete;
    Logger& operator=(const Logger&) = delete;

    // Get singleton instance
    static Logger& getInstance();

    // Logging functions
    void setLogLevel(LogLevel level);
    void setOutputStream(std::ostream& out);
    void log(LogLevel level, const std::string& message);

    void debug(const std::string& message);
    void info(const std::string& message);
    void warn(const std::string& message);
    void error(const std::string& message);
    void fatal(const std::string& message);
};

void debug(const std::string& message);
void info(const std::string& message);
void warn(const std::string& message);
void error(const std::string& message);
void fatal(const std::string& message);

#endif // LOGGER_H
