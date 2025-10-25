cmake_minimum_required(VERSION 3.16)

project(FaceComparator VERSION 0.1 LANGUAGES CXX)

# Qt otomatik araçları
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)

# C++ standardı
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Build tipine göre DLL seçimi
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(OPENCV_DLL_NAME "opencv_world4120d.dll")
else()
    set(OPENCV_DLL_NAME "opencv_world4120.dll")
endif()

# Qt bulun
find_package(QT NAMES Qt6 Qt5 REQUIRED COMPONENTS Widgets Core Gui)
find_package(Qt${QT_VERSION_MAJOR} REQUIRED COMPONENTS Widgets Core Gui)

# OpenCV bulun
set(OpenCV_DIR "D:/opencv/opencv/build")
find_package(OpenCV REQUIRED)

# ONNX Runtime hazır build yolu
set(ONNXRUNTIME_DIR "D:/onnxruntime-win-x64-1.23.1")

# Proje kaynakları
set(PROJECT_SOURCES
    main.cpp
    mainwindow.cpp
    mainwindow.h
    mainwindow.ui
)

# Qt6 veya Qt5'e göre executable
if(${QT_VERSION_MAJOR} GREATER_EQUAL 6)
    qt_add_executable(FaceComparator
        MANUAL_FINALIZATION
        ${PROJECT_SOURCES}
        resources.qrc
    )
else()
    if(ANDROID)
        add_library(FaceComparator SHARED ${PROJECT_SOURCES})
    else()
        add_executable(FaceComparator ${PROJECT_SOURCES})
    endif()
endif()

# Include path'ler
target_include_directories(FaceComparator PRIVATE
    ${ONNXRUNTIME_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
)

# Link kütüphaneleri
target_link_libraries(FaceComparator PRIVATE
    Qt${QT_VERSION_MAJOR}::Widgets
    Qt${QT_VERSION_MAJOR}::Core
    Qt${QT_VERSION_MAJOR}::Gui
    "${ONNXRUNTIME_DIR}/lib/onnxruntime.lib"
    ${OpenCV_LIBS}
)

# Sadece gerekli DLL'leri kopyala
add_custom_command(TARGET FaceComparator POST_BUILD
    # ONNX Runtime DLL
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${ONNXRUNTIME_DIR}/lib/onnxruntime.dll"
        $<TARGET_FILE_DIR:FaceComparator>

    # OpenCV DLL
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${OpenCV_DIR}/x64/vc16/bin/${OPENCV_DLL_NAME}"
        $<TARGET_FILE_DIR:FaceComparator>

    # OpenCV ek DLL'ler (gerekli olabilir)
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${OpenCV_DIR}/x64/vc16/bin/opencv_videoio_ffmpeg4120_64.dll"
        $<TARGET_FILE_DIR:FaceComparator>

    COMMENT "Gerekli DLL'ler kopyalanıyor..."
)

# iOS/macOS bundle ayarları
if(${QT_VERSION} VERSION_LESS 6.1.0)
  set(BUNDLE_ID_OPTION MACOSX_BUNDLE_GUI_IDENTIFIER com.example.FaceComparator)
endif()

set_target_properties(FaceComparator PROPERTIES
    ${BUNDLE_ID_OPTION}
    MACOSX_BUNDLE_BUNDLE_VERSION ${PROJECT_VERSION}
    MACOSX_BUNDLE_SHORT_VERSION_STRING ${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}
    MACOSX_BUNDLE TRUE
    WIN32_EXECUTABLE TRUE
)

# Install ayarları
include(GNUInstallDirs)
install(TARGETS FaceComparator
    BUNDLE DESTINATION .
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

# Qt6 finalize
if(QT_VERSION_MAJOR EQUAL 6)
    qt_finalize_executable(FaceComparator)
endif()

