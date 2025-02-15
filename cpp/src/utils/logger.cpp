// Logger.cpp
#include "logger.hpp"

// Singleton instance function
Logger& Logger::getInstance() {
    static Logger instance;
    return instance;
}

// Private constructor
Logger::Logger() : currentLevel(LogLevel::INFO), outputStream(&std::cout) {}

// Set log level
void Logger::setLogLevel(LogLevel level) {
    currentLevel = level;
}

// Set output stream (console or file)
void Logger::setOutputStream(std::ostream& out) {
    outputStream = &out;
}

// Log function
void Logger::log(LogLevel level, const std::string& message) {
    if (level >= currentLevel) {
        std::lock_guard<std::mutex> lock(logMutex);

        std::string levelStr;
        switch (level) {
            case LogLevel::DEBUG: levelStr = "DEBUG"; break;
            case LogLevel::INFO: levelStr = "INFO"; break;
            case LogLevel::WARN: levelStr = "WARN"; break;
            case LogLevel::ERROR: levelStr = "ERROR"; break;
            case LogLevel::FATAL: levelStr = "FATAL"; break;
        }

        (*outputStream) << "[" << levelStr << "] " << message << std::endl;
    }
}

// Convenience logging methods
void Logger::debug(const std::string& message) { log(LogLevel::DEBUG, message); }
void Logger::info(const std::string& message) { log(LogLevel::INFO, message); }
void Logger::warn(const std::string& message) { log(LogLevel::WARN, message); }
void Logger::error(const std::string& message) { log(LogLevel::ERROR, message); }
void Logger::fatal(const std::string& message) { log(LogLevel::FATAL, message); }

// Convenience logging methods
Logger& logger = Logger::getInstance();
void debug(const std::string& message) { logger.log(LogLevel::DEBUG, message); }
void info(const std::string& message) { logger.log(LogLevel::INFO, message); }
void warn(const std::string& message) { logger.log(LogLevel::WARN, message); }
void error(const std::string& message) { logger.log(LogLevel::ERROR, message); }
void fatal(const std::string& message) { logger.log(LogLevel::FATAL, message); }
