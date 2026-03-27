cmake_minimum_required(VERSION 3.10)
project(spot_detector_rknn LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---------- RKNN Runtime ----------
if(NOT DEFINED RKNN_API_PATH)
    set(RKNN_API_PATH "/usr/local" CACHE PATH "Path to RKNN API (contains include/rknn_api.h and lib/)")
endif()

find_path(RKNN_INCLUDE_DIR rknn_api.h
    PATHS ${RKNN_API_PATH}/include/rknn ${RKNN_API_PATH}/include
          /usr/include/rknn /usr/include /usr/local/include/rknn /usr/local/include
)
find_library(RKNN_LIBRARY rknnrt
    PATHS ${RKNN_API_PATH}/lib64 ${RKNN_API_PATH}/lib
          /usr/lib64 /usr/lib /usr/local/lib64 /usr/local/lib
)

if(NOT RKNN_INCLUDE_DIR OR NOT RKNN_LIBRARY)
    message(FATAL_ERROR "RKNN API not found. Set -DRKNN_API_PATH=<path>")
endif()

message(STATUS "RKNN include: ${RKNN_INCLUDE_DIR}")
message(STATUS "RKNN library: ${RKNN_LIBRARY}")

# ---------- OpenCV ----------
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs highgui)
message(STATUS "OpenCV: ${OpenCV_VERSION}")

# ---------- Library ----------
add_library(spot_detector STATIC
    src/spot_detector.cpp
)
target_include_directories(spot_detector PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${RKNN_INCLUDE_DIR}
)
target_link_libraries(spot_detector PUBLIC
    ${RKNN_LIBRARY}
    ${OpenCV_LIBS}
)

# ---------- GStreamer (optional, for streaming) ----------
find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
    pkg_check_modules(GSTREAMER gstreamer-1.0 gstreamer-app-1.0)
endif()

# ---------- Single-image CLI ----------
add_executable(spot_detect src/main.cpp)
target_link_libraries(spot_detect PRIVATE spot_detector)

# ---------- Real-time streaming ----------
if(GSTREAMER_FOUND)
    message(STATUS "GStreamer found — building spot_stream")
    add_executable(spot_stream src/main_stream.cpp)
    target_include_directories(spot_stream PRIVATE ${GSTREAMER_INCLUDE_DIRS})
    target_link_libraries(spot_stream PRIVATE
        spot_detector
        ${GSTREAMER_LIBRARIES}
        pthread
    )
else()
    message(STATUS "GStreamer not found — skipping spot_stream (install gstreamer-1.0 and gstreamer-app-1.0 to enable)")
endif()
