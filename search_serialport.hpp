#include <iostream>
#include <string>
#include <vector>
#include <dirent.h>
#include <regex>

// Function to search for existing serial port names matching a pattern
std::vector<std::string> findSerialPorts(const std::string& pattern) {
    std::vector<std::string> serialPorts;

    DIR *dir;
    struct dirent *ent;
    if ((dir = opendir("/dev")) != nullptr) {
        // Regular expression pattern to match the desired serial port names
        std::regex regexPattern(pattern);

        // Iterate through all entries in the /dev directory
        while ((ent = readdir(dir)) != nullptr) {
            std::string deviceName = std::string(ent->d_name);

            // If the entry matches the pattern, add it to the list of serial ports
            if (std::regex_match(deviceName, regexPattern)) {
                serialPorts.push_back("/dev/" + deviceName);
            }
        }
        closedir(dir);
    } else {
        // Could not open directory
        std::cerr << "Error opening directory /dev" << std::endl;
    }

    return serialPorts;
}