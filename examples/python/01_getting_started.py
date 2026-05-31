"""ELIPS getting started — open, place, search, filter, EQL.

Run from the repo root after building the bindings:
    cmake -S . -B build -G Ninja -DELIPS_BUILD_PYTHON=ON
    cmake --build build --target elips_pymodule
    PYTHONPATH=bindings/python python3 examples/python/01_getting_started.py
"""

import elips


def main() -> None:
    with elips.open(":memory:", dimension=3, metric="cosine") as db:
        docs = db.vault("documents")
        docs.place([1.0, 0.0, 0.0], {"title": "alpha", "year": 2024})
        docs.place([0.0, 1.0, 0.0], {"title": "beta", "year": 2019})
        docs.place([0.9, 0.1, 0.0], {"title": "gamma", "year": 2023})

        print("nearest to [1, 0, 0]:")
        for hit in docs.seek([1.0, 0.0, 0.0], top=3):
            print(f"  {hit.data['title']:6} distance={hit.distance:.4f}")

        recent = elips.Filter().field("year").gte(2023)
        print("\nfiltered (year >= 2023):")
        for hit in docs.seek([1.0, 0.0, 0.0], top=3, where=recent):
            print(f"  {hit.data['title']} ({hit.data['year']})")

        print("\nvia EQL:")
        rows = db.query(
            "seek in documents nearest $q top 2 project title yield",
            bindings={"q": [1.0, 0.0, 0.0]},
        )
        print("  ", [r.data["title"] for r in rows])


if __name__ == "__main__":
    main()
