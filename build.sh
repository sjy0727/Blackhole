conan install . --output-folder=build --build=missing
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_TOOLCHAIN_FILE="build/conan_toolchain.cmake"
cmake --build build