
#include <iostream>
#include <string>
#include <cstring>
#include <sstream>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <vector>
#include <map>

#include <util/dbscan.h>
#include <util/assertion.h>

/*
 *  Given data, density, and epsilon perform clustering
 */

// Errors
const int CommandLineArgumentError = 1;

// Command Line Argument Meta-data
const int RequiredArgumentCount = 5;

// Command line arguments
std::string programName;
std::string inputFile;
std::string outputFile;
int density;
double epsilon;


void usage() {
    std::cerr << "usage: " << programName << " <input> <output> <density> <epsilon>" << std::endl << std::endl;

    std::cerr << "\tinput" << std::endl;
    std::cerr << "\t\tfilename containing training data" << std::endl << std::endl;
    
    std::cerr << "\toutput" << std::endl;
    std::cerr << "\t\toutput filename containing containing meta-data about clustering results" << std::endl;
    std::cerr << "\t\t, data is partitioned into additional files of the form <output>-1, ... , <output>-N" << std::endl << std::endl;
    
    std::cerr << "\tdensity" << std::endl;
    std::cerr << "\t\tnumber of points that have to be in the neighborhood of a point in order to be considered a cluster point" << std::endl << std::endl;
    
    std::cerr << "\tepsilon" << std::endl;
    std::cerr << "\t\thow close another point has to be in order to considered in the neighborhood" << std::endl << std::endl;
}

void parseArguments(int argc, char** argv) {
    programName = std::string(argv[0]);
    if(argc != RequiredArgumentCount) {
        std::cerr << "not enough arguments... " << argc << std::endl;
        usage();
        exit(CommandLineArgumentError);
    }

    inputFile = argv[1];
    outputFile = argv[2];

    try {
        size_t pos = 0;
        density = std::stoi(argv[3],&pos);
        if(pos != strlen(argv[3])) {
            throw std::runtime_error("oops");
        }
    } catch(...) {
        std::cerr << "Could not convert the command line value " << argv[3] << " to an integer for density value." << std::endl;
        exit(CommandLineArgumentError);
    }

    try {
        size_t pos = 0;
        epsilon = std::stod(argv[4],&pos);
        if(pos != strlen(argv[4])) {
            throw std::runtime_error("oops");
        }
    } catch(...) {
        std::cerr << "Could not convert the command line value " << argv[4] << " to a double for epsilon value." << std::endl;
        exit(CommandLineArgumentError);
    }
}

struct Data {
    std::vector<std::vector<float>> data;
    int width;
    int N;

    Data() {
        width = 0;
        N = 0;
    }


    Data(Data && other) {
        *this = other;
    }

    Data(const Data & other) {
        *this = other;
    }

    void operator=(Data && other) {
        width = other.width;
        N = other.N;
        data.swap(other.data);
    }

    void operator=(const Data & other) {
        width = other.width;
        N = other.N;
        data.clear();
        data.insert(data.end(),other.data.begin(),other.data.end());
    }

    std::string toString() {
        std::stringstream ss;

        ss << N << std::endl;
        ss << width << std::endl;
        int pos = 0;

        for(auto it = data.begin(); it != data.end(); ++it) {
            int j = 0;
            for(auto vit = it->begin(); vit != it->end(); ++vit) {
                if(*vit != 0) {
                    ss << j << ":" << *vit << " ";
                }
                ++j;
            }
            ss << std::endl;
        }
        return ss.str();
    }
    
};

std::vector<std::string> split(const std::string & line, char delim) {
    std::vector<std::string> result;
    std::stringstream ss(line);
    std::string item;
    while(std::getline(ss,item,delim) ) {
        result.push_back(item);
    }
    return result;
}

int MaxN = 50000;

Data readData() {
    Data result;
    std::ifstream input(inputFile);

    int maxID = 0;
    int N = 0;

    std::vector<int>                labels;
    std::vector<std::vector<float>> data;
    std::vector<std::vector<int>>   indices;

    std::string line;
    while(std::getline(input,line)) {

        std::vector<float> lineData;
        std::vector<int> lineIndices;
        std::stringstream ss;
        ss << line;

        ss >> line;
        labels.push_back(std::stoi(line));
        
        // qid
        ss >> line;

        while(ss >> line) {

            // Skip comments
            if(line[0] == '#') {
                break;
            }
            std::vector<std::string> tokens = split(line,':');
            Assert(tokens.size(),2,true);
            int id = std::stoi(tokens[0]);
            float value = std::stof(tokens[1]);

            if(id > maxID) {
                maxID = id;
            }

            lineIndices.push_back(id);
            lineData.push_back(value);
        }

        data.push_back(lineData);
        indices.push_back(lineIndices);
        ++N;
        if(N >= MaxN) break;
    }
    ++maxID;
    const int numElements = maxID * N;

    result.N = N;
    result.width = maxID;
    result.data = std::vector<std::vector<float>>(N,std::vector<float>(maxID,0.0));

    for(int i = 0 ; i < data.size(); ++i) {
        for(int j = 0; j < indices[i].size();++j) {
            uint idx = indices[i][j];
            float value = data[i][j];
            AssertLessThan(idx,result.data[i].size());
            result.data[i][idx] = value;
        }
    }
    return result;
}

 int main(int argc, char** argv) {

    Math::Manhattan met;
    parseArguments(argc,argv);

    std::cout << "running dbscan with arguments" << std::endl;
    std::cout << "\tinput=" << inputFile << std::endl;
    std::cout << "\toutput=" << outputFile << std::endl;
    std::cout << "\tk=" << density << std::endl;
    std::cout << "\teps=" << epsilon << std::endl;

    std::cout << "loading data" << std::flush;
    Data data = readData();
    std::cout << " ...data loaded" << std::flush;

    std::cout << " ...initializing algorithm" << std::flush;
    DBSCANCPU alg(data.data,density,epsilon,met);
    std::cout << " ...running algorithm" << std::flush;
    std::cout << " ...finding cluster points" << std::flush;
    alg.locateClusterPoints();
    std::cout << " ...merging clusters" << std::flush;
    alg.mergeClusters();
    std::cout << " ...merging noise" << std::flush;
    alg.aggregateNoise();

    std::cout << " ...saving results" << std::flush;

    std::map<int,std::vector<int>> clusters;
    auto ids = alg.getIDs();
     for(int i = 0 ; i < data.N; ++i) {
         int idx = ids[i];
         auto ptr = clusters.find(idx);
         if(ptr != clusters.end()) {
             clusters[idx].push_back(i);
         }else {
             clusters[idx] = {i};
         }
     }

     int clusterCount = 0;
     for(auto elem : clusters ) {
         if(elem.second.size() >= density) {
             clusterCount++;
         }
     }

     std::ofstream output(outputFile);
     output << clusterCount << std::endl;
     for(auto elem : clusters ) {
         if(elem.second.size() >= density) {
            output << elem.first << ": " << elem.second.size() << std::endl;
            for(auto idx : elem.second) {
                output << "\t" << idx << std::endl;
            }
         }
     }
     output.close();
     std::cout << " ...results saved" << std::endl;

     return 0;
 }
