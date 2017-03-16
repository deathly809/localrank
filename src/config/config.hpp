
#ifndef CONFIG_HPP_
#define CONFIG_HPP_

#include <map>
#include <sstream>

template<typename F, typename T>
T convert(F from) {
    T result;
    std::stringstream ss;
    ss << from;
    ss >> result;
}

class Configuration {
    private:
    
        std::map<std::string,std::string> entries;
    
    public:
        
        Configuration();
        
        // Parse command line arguments
        ParseArgs(int argc, char* argv[]);
        
        // Parse a file
        ParseFile(std::string filename);
        
        // Add a new value
        template <typename T>
        void Insert(std::string key, T valuee);
        
        // Try to convert a value to the underlying representation 
        template <typename T>
        T Get<(std::string key) {
            auto needle = entries.find(key);
            if(needle == entries.end()) {
                throw std::runtime_exception("Key does not exist in configuration");
            }
            
            return convert<std::string,T>(needle->second);
        }
        
        // Reports if the configuration key exists
        bool HasKey(std::string key) {
            return entries.find(key) != entries.end();
        }
        
        // Remove from the configuration
        void Remove(std::string key);
};

template<typename K,typename V>
struct KeyValuePair {
    K key;
    V value;
};

void Configuration Preload(KeyValuePair<std::string,std::string> entries) {
    Configuration c;
    for( auto e : entries ) {
        c.Insert(e.key,e.value);
    }
    return c;
}

#endif 