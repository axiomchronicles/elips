// libFuzzer entry point for the WAL record parser -- the same surface F2 was
// about (length-prefixed reads ahead of the CRC check).
//
// Build and run:
//   cmake -S . -B build-fuzz -DELIPS_BUILD_FUZZERS=ON \
//     -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_BUILD_TYPE=Debug
//   cmake --build build-fuzz --target elips_fuzz_wal
//   ./build-fuzz/elips_fuzz_wal -max_total_time=60
//
// On macOS, Apple's CommandLineTools clang ships no libFuzzer runtime; point
// CMAKE_CXX_COMPILER at a full LLVM install (e.g. /opt/homebrew/opt/llvm).
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>

#include "elips/storage/WAL.hpp"

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data,
                                     std::size_t size) {
    // replay() takes a path, so stage the input in a temp file per iteration.
    static const std::filesystem::path path =
        std::filesystem::temp_directory_path() / "elips_fuzz_wal.log";
    {
        std::ofstream out(path, std::ios::binary | std::ios::trunc);
        out.write(reinterpret_cast<const char*>(data),
                  static_cast<std::streamsize>(size));
    }
    // Any exception escaping here is itself a finding: recovery must treat
    // malformed input as a corrupt tail, not propagate.
    const auto entries = elips::WAL::replay(path);
    return static_cast<int>(entries.size() & 0);
}
