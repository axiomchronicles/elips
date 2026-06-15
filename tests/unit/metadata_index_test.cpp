#include <gtest/gtest.h>

#include <set>
#include <string>
#include <vector>

#include "elips/domain/Record.hpp"
#include "elips/domain/RecordID.hpp"
#include "elips/metadata/MetadataIndex.hpp"

namespace {

TEST(MetadataIndexTest, ExactCandidatesIntersectEqualityConstraints) {
    elips::MetadataIndex index;
    const auto alpha_design = elips::RecordID::generate();
    const auto alpha_ops = elips::RecordID::generate();
    const auto beta_design = elips::RecordID::generate();

    index.insert(alpha_design,
                 {{"tenant", std::string{"alpha"}},
                  {"kind", std::string{"design"}}});
    index.insert(alpha_ops,
                 {{"tenant", std::string{"alpha"}},
                  {"kind", std::string{"ops"}}});
    index.insert(beta_design,
                 {{"tenant", std::string{"beta"}},
                  {"kind", std::string{"design"}}});

    const auto filter = elips::Filter()
                            .field("tenant")
                            .equals(std::string{"alpha"})
                            .field("kind")
                            .equals(std::string{"design"});

    const auto candidates = index.exact_candidates(filter);
    ASSERT_TRUE(candidates.has_value());
    EXPECT_EQ(*candidates, std::set<elips::RecordID>{alpha_design});
}

TEST(MetadataIndexTest, ExactCandidatesSupportInSetAndRemoval) {
    elips::MetadataIndex index;
    const auto design = elips::RecordID::generate();
    const auto ops = elips::RecordID::generate();
    const auto finance = elips::RecordID::generate();

    const elips::Payload design_payload{{"kind", std::string{"design"}}};
    const elips::Payload ops_payload{{"kind", std::string{"ops"}}};
    const elips::Payload finance_payload{{"kind", std::string{"finance"}}};

    index.insert(design, design_payload);
    index.insert(ops, ops_payload);
    index.insert(finance, finance_payload);

    const auto filter = elips::Filter().field("kind").one_of(
        std::vector<elips::MetaValue>{std::string{"design"},
                                      std::string{"ops"}});

    const auto before_remove = index.exact_candidates(filter);
    ASSERT_TRUE(before_remove.has_value());
    EXPECT_EQ(*before_remove, (std::set<elips::RecordID>{design, ops}));

    index.remove(ops, ops_payload);

    const auto after_remove = index.exact_candidates(filter);
    ASSERT_TRUE(after_remove.has_value());
    EXPECT_EQ(*after_remove, std::set<elips::RecordID>{design});
}

TEST(MetadataIndexTest, NonExactFiltersDoNotProduceCandidates) {
    elips::MetadataIndex index;
    index.insert(elips::RecordID::generate(),
                 {{"kind", std::string{"design"}},
                  {"title", std::string{"alpha design note"}}});

    const auto contains_filter =
        elips::Filter().field("title").contains("design");
    EXPECT_FALSE(index.exact_candidates(contains_filter).has_value());

    const auto disjunctive_filter =
        elips::Filter().field("kind").equals(std::string{"design"}).or_(
            elips::Filter().field("kind").equals(std::string{"ops"}));
    EXPECT_FALSE(index.exact_candidates(disjunctive_filter).has_value());

    EXPECT_FALSE(
        index.exact_candidates(elips::Filter::not_(elips::Filter().field("kind").equals(
            std::string{"design"})))
            .has_value());
}

}  // namespace
