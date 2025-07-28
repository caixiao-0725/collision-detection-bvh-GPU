#include <stdio.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include "Mesh.h"
#include "DeviceHostVector.h"
#include "bv.h"
#include "atomicFunctions.cuh"

#include "lbvh.h"

#include <unordered_set>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <string>
#include <iomanip>
#include <fstream>
#include <sstream>
#include "common.h"

// Platform-specific filesystem includes
#ifdef __has_include
#if __has_include(<filesystem>)
#include <filesystem>
namespace fs = std::filesystem;
#elif __has_include(<experimental/filesystem>)
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;
#else
#error "filesystem not available"
#endif
#else
#include <filesystem>
namespace fs = std::filesystem;
#endif

using namespace Lczx;


std::vector<BOX> load_aabb_file(const std::string& filepath) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file " << filepath << std::endl;
        return {};
    }

    file.seekg(0, std::ios::end);
    size_t file_size = file.tellg();
    file.seekg(0, std::ios::beg);

    size_t num_aabbs = file_size / (6 * sizeof(float));

    std::vector<BOX> aabbs;
    aabbs.reserve(num_aabbs);

    for (size_t i = 0; i < num_aabbs; i++) {
        float data[6];
        file.read(reinterpret_cast<char*>(data), 6 * sizeof(float));

        if (file.gcount() != 6 * sizeof(float)) {
            std::cerr << "Error: Incomplete read from file " << filepath << std::endl;
            break;
        }
        BOX temp;
        temp._min = vec3f(data[0], data[1], data[2]);
        temp._max = vec3f(data[3], data[4], data[5]);
        aabbs.emplace_back(temp);
    }

    file.close();
    return aabbs;
}

std::vector<int> get_frame_numbers(const std::string& dataset_path) {
    std::vector<int> frame_numbers;

    try {
        for (const auto& entry : fs::directory_iterator(dataset_path)) {
            if (entry.is_regular_file()) {
                std::string filename = entry.path().filename().string();
                if (filename.find("mesh_triangle_aabb_frame_") == 0) {
                    // Extract frame number from filename
                    size_t start = filename.find("_frame_") + 7;
                    size_t end = filename.find(".bin");
                    if (end != std::string::npos) {
                        std::string frame_str = filename.substr(start, end - start);
                        int frame_num = std::stoi(frame_str);
                        frame_numbers.push_back(frame_num);
                    }
                }
            }
        }
    }
    catch (const fs::filesystem_error& ex) {
        std::cerr << "Error accessing directory " << dataset_path << ": " << ex.what() << std::endl;
    }

    std::sort(frame_numbers.begin(), frame_numbers.end());
    return frame_numbers;
}

void loadObj(std::string filename, std::vector<vec3f>& position, std::vector<vec3i>& face,std::vector<vec2i>& edge) {
    std::ifstream in;
    in.open(filename, std::ifstream::in);
    if (in.fail())
        return;
    std::string line;
    while (!in.eof())
    {
        std::getline(in, line);
        std::istringstream iss(line.c_str());
        char trash;
        if (!line.compare(0, 2, "v "))
        {
            iss >> trash;
            vec3f v;
            for (int i = 0; i < 3; i++)
                iss >> v.raw[i];
            position.push_back(v);
        }
        else if (!line.compare(0, 2, "f "))
        {
            int iuv, idx;
            iss >> trash;
            vec3i f;
			int count = 0;
            std::string vertex;
            while (iss >> vertex)
            {
                std::istringstream vss(vertex);
                std::string index;
                std::getline(vss, index, '/');
                std::istringstream(index) >> idx;
				f.raw[count] = idx-1;
                count++;
            }
            face.push_back(f);
        }
    }

    std::set<std::pair<int, int>> edges;
    for (int f = 0; f < face.size();f++) {
        for (int i = 0;i < 3;i++) {
			int v1 = face[f].raw[i];
			int v2 = face[f].raw[(i + 1) % 3];
			if (v1 > v2) std::swap(v1, v2);
			edges.insert(std::make_pair(v1, v2));
        }
    }

	for (const auto& e : edges) {
		edge.push_back(vec2i(e.first, e.second));
	}

    std::cerr << "[Mesh] vert: " << position.size() << "   face: " << face.size() << std::endl;
}

struct TestResult {
    int frame;
    float time;
    size_t mesh_count;
    size_t collider_count;
    float collider_build_time;
    float mesh_build_time;
    float query_time;
    float self_query_time;
    size_t query_contacts;
    size_t self_query_contacts;
    bool query_matches_ground_truth;
    bool self_query_matches_ground_truth;
};

void save_results_to_csv(const std::string& dataset_name, const std::vector<TestResult>& results) {
    std::string output_dir = get_asset_path() + "statistic/";

    fs::create_directories(output_dir);

    std::string csv_path = output_dir + dataset_name + ".csv";
    std::ofstream csv_file(csv_path);

    if (!csv_file.is_open()) {
        std::cerr << "Error: Cannot create CSV file " << csv_path << std::endl;
        return;
    }

    csv_file << "frame,time,mesh_count,collider_count,"
        << "collider_build_time,mesh_build_time,query_time,self_query_time,"
        << "query_contacts,self_query_contacts,"
        << "query_matches_ground_truth,self_query_matches_ground_truth\n";

    for (const auto& result : results) {
        csv_file << result.frame << ","
            << std::fixed << std::setprecision(2) << result.time << ","
            << result.mesh_count << "," << result.collider_count << ","
            << std::fixed << std::setprecision(3)
            << result.collider_build_time << "," << result.mesh_build_time << ","
            << result.query_time << "," << result.self_query_time << ","
            << result.query_contacts << "," << result.self_query_contacts << ","
            << (result.query_matches_ground_truth ? "true" : "false") << ","
            << (result.self_query_matches_ground_truth ? "true" : "false") << "\n";
    }

    csv_file.close();
    std::cout << "Results saved to " << csv_path << std::endl;
}

void test_obj() {
    std::string mesh_file = get_asset_path() + "models/0.obj";
	std::vector<vec3f> position;
	std::vector<vec3i> face;
	std::vector<vec2i> edge;
	loadObj(mesh_file, position, face, edge);
	DeviceHostVector<vec3f> d_position;
	d_position.Allocate(position.size());
	d_position.SetDeviceHost(position.size(), position.data());
	DeviceHostVector<vec3i> d_face;
	d_face.Allocate(face.size());
	d_face.SetDeviceHost(face.size(), face.data());
	DeviceHostVector<vec2i> d_edge;
	d_edge.Allocate(edge.size());
	d_edge.SetDeviceHost(edge.size(), edge.data());
	Bvh A;
	A._type = 5;
	A.setup(d_face.GetSize(), d_face.GetSize(), d_face.GetSize() - 1);
	A.build(d_position, d_face);
    A.queryEF(d_position, d_edge, d_face, d_edge.GetSize());
	//std::cout << "AABBs built for mesh with " << face.size() << " faces." << std::endl;
}

void test_frame(const std::string& dataset_name, int frame_num) {

    std::string asset_path = get_asset_path();
    std::string mesh_file = asset_path + dataset_name + "/mesh_triangle_aabb_frame_" +
        std::string(6 - std::to_string(frame_num).length(), '0') + std::to_string(frame_num) + ".bin";
    std::string collider_file = asset_path + dataset_name + "/moving_collider_triangle_aabb_frame_" +
        std::string(6 - std::to_string(frame_num).length(), '0') + std::to_string(frame_num) + ".bin";

    auto mesh_aabbs = load_aabb_file(mesh_file);
    auto collider_aabbs = load_aabb_file(collider_file);

    std::cout << "  Mesh AABBs: " << mesh_aabbs.size() << ", Collider AABBs: " << collider_aabbs.size() << std::endl;
    
    DeviceHostVector<AABB> d_mesh_aabbs;
    DeviceHostVector<AABB> d_collider_aabbs;
    d_mesh_aabbs.Allocate(mesh_aabbs.size());
    d_collider_aabbs.Allocate(collider_aabbs.size());
    d_mesh_aabbs.SetDeviceHost(mesh_aabbs.size(), mesh_aabbs.data());
    d_collider_aabbs.SetDeviceHost(collider_aabbs.size(), collider_aabbs.data());
    
    Bvh A;
    A._type = 4;
    A.setup(mesh_aabbs.size(), mesh_aabbs.size(), mesh_aabbs.size() - 1);


    std::cout << "  Testing culbvh..." << std::endl;


    A.build(d_mesh_aabbs.GetDevice());

    
    // Query between collider and mesh
    A.query(d_mesh_aabbs.GetDevice(), d_mesh_aabbs.GetSize(), true);
    //A.query(d_collider_aabbs.GetDevice(), d_collider_aabbs.GetSize(),false);

    return ;
}



int main() {
    //test_obj();
    test_frame("dance",650);
    return 0;
}

