#include <gtest/gtest.h>

#include <vector>

#include "elips/Config.hpp"
#include "elips/domain/Errors.hpp"
#include "elips/domain/RecordID.hpp"
#include "elips/index_engine/IndexFlat.hpp"
#include "elips/index_engine/IndexIVFFlat.hpp"
#include "elips/index_engine/IndexIVFPQ.hpp"
#include "elips/index_engine/IndexIVFScalarQuantizer.hpp"

namespace {

using elips::ConfigError;
using elips::DimensionMismatch;
using elips::DimensionMismatchError;
using elips::IndexFlat;
using elips::IndexIVFFlat;
using elips::IndexIVFPQ;
using elips::IndexIVFScalarQuantizer;
using elips::Metric;
using elips::NotTrainedError;
using elips::RecordID;

// --- dimension mismatch -> DimensionMismatchError --------------------------

TEST(TypedErrors, FlatAddWrongDimensionThrowsDimensionMismatch) {
    IndexFlat idx(Metric::euclidean, 4);
    EXPECT_THROW(idx.add_with_ids(RecordID::generate(),
                                  std::vector<float>{1.0F, 2.0F, 3.0F}),
                 DimensionMismatchError);
}

TEST(TypedErrors, FlatSearchWrongDimensionThrowsDimensionMismatch) {
    IndexFlat idx(Metric::euclidean, 4);
    idx.add_with_ids(RecordID::generate(),
                     std::vector<float>{1.0F, 2.0F, 3.0F, 4.0F});
    EXPECT_THROW((void)idx.search(std::vector<float>{1.0F, 2.0F}, 1),
                 DimensionMismatchError);
}

TEST(TypedErrors, DimensionMismatchErrorIsAliasOfDimensionMismatch) {
    // The *Error spelling is the same type: a throw of one is caught by the
    // other, matching the Python alias and keeping pybind translation stable.
    IndexFlat idx(Metric::euclidean, 4);
    EXPECT_THROW(idx.add_with_ids(RecordID::generate(),
                                  std::vector<float>{1.0F}),
                 DimensionMismatch);
}

// --- IVF add/search before train -> NotTrainedError ------------------------

TEST(TypedErrors, IVFFlatAddBeforeTrainThrowsNotTrained) {
    IndexIVFFlat idx(Metric::euclidean, 4, 8);
    EXPECT_FALSE(idx.is_trained());
    EXPECT_THROW(idx.add_with_ids(RecordID::generate(),
                                  std::vector<float>{1.0F, 2.0F, 3.0F, 4.0F}),
                 NotTrainedError);
}

TEST(TypedErrors, IVFFlatSearchBeforeTrainThrowsNotTrained) {
    IndexIVFFlat idx(Metric::euclidean, 4, 8);
    EXPECT_THROW(
        (void)idx.search(std::vector<float>{1.0F, 2.0F, 3.0F, 4.0F}, 1),
        NotTrainedError);
}

TEST(TypedErrors, IVFPQAndSQAddBeforeTrainThrowNotTrained) {
    IndexIVFPQ pq(Metric::euclidean, 8, 4, 4);
    IndexIVFScalarQuantizer sq(Metric::euclidean, 8, 4);
    EXPECT_THROW(
        pq.add_with_ids(RecordID::generate(), std::vector<float>(8, 1.0F)),
        NotTrainedError);
    EXPECT_THROW(
        sq.add_with_ids(RecordID::generate(), std::vector<float>(8, 1.0F)),
        NotTrainedError);
}

TEST(TypedErrors, NotTrainedErrorStillCatchableAsConfigError) {
    // Backward compat: before-train use has always surfaced as ConfigError.
    IndexIVFFlat idx(Metric::euclidean, 4, 8);
    EXPECT_THROW(idx.add_with_ids(RecordID::generate(),
                                  std::vector<float>{1.0F, 2.0F, 3.0F, 4.0F}),
                 ConfigError);
}

}  // namespace
