from src.fetch_papers import find_dataset_paper


def test_isruc_sleep_dataset_anchor_uses_dataset_access_site():
    for dataset_name in ["ISRUC", "ISRUC-SLEEP", "ISRUC-Sleep", "ISRUC-S1", "ISRUC-S3"]:
        result = find_dataset_paper(
            dataset_name,
            full_name="ISRUC-Sleep: A comprehensive public dataset for sleep researchers",
            modalities=["PSG (polysomnography)"],
        )

        assert result["found"]
        assert result["title"] == "ISRUC-Sleep: A comprehensive public dataset for sleep researchers"
        assert result["doi"] == "10.1016/j.cmpb.2015.10.013"
        assert result["paper_url"] == "https://sleeptight.isr.uc.pt/"
        assert result["_candidate_ranking"][0]["source"] == "curated dataset anchor"
