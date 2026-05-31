#include <gtest/gtest.h>

#include <filesystem>
#include <random>

#include "elips/elips.hpp"
#include "elips/kernel/LockManager.hpp"

namespace {

namespace fs = std::filesystem;

TEST(SingleWriterTest, SecondWriterFailsWithLockConflict) {
    std::random_device rd;
    const fs::path dir =
        fs::temp_directory_path() / ("elips_lock_" + std::to_string(rd()));

    auto first = elips::open(dir.string(), elips::Config{}.dimension(2));
    EXPECT_THROW((void)elips::open(dir.string(), elips::Config{}.dimension(2)),
                 elips::LockConflict);

    first->close();
    first.reset();  // release the lock

    // After the first writer releases, a new writer can open.
    EXPECT_NO_THROW({
        auto second = elips::open(dir.string());
        second->close();
    });

    std::error_code ec;
    fs::remove_all(dir, ec);
}

}  // namespace
