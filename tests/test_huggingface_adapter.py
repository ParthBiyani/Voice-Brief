from __future__ import annotations

from datetime import datetime, timedelta, timezone

import httpx
import pytest
import respx

from voicebrief.sources.base import SourceConfig
from voicebrief.sources.huggingface import HuggingFaceAdapter

MODELS_URL = "https://huggingface.co/api/models"


def record(repo_id: str, **over) -> dict:
    base = {
        "id": repo_id,
        "author": repo_id.split("/")[0],
        "lastModified": (datetime.now(timezone.utc) - timedelta(days=1))
        .isoformat()
        .replace("+00:00", "Z"),
        "downloads": 10_000,
        "likes": 50,
        "pipeline_tag": "text-generation",
        "library_name": "transformers",
        "tags": ["llama", "conversational", "license:apache-2.0", "arxiv:2401.00001", "en"],
    }
    return {**base, **over}


def adapter(entity: str = "models") -> HuggingFaceAdapter:
    return HuggingFaceAdapter(
        SourceConfig(
            slug="hf",
            name="HF",
            endpoint=MODELS_URL,
            default_topics=["machine-learning"],
            config={"kind": entity, "limit": 50},
        )
    )


@pytest.fixture
def since() -> datetime:
    return datetime.now(timezone.utc) - timedelta(days=14)


async def run(payload, entity="models", since=None):
    since = since or datetime.now(timezone.utc) - timedelta(days=14)
    respx.get(MODELS_URL).mock(return_value=httpx.Response(200, json=payload))
    async with httpx.AsyncClient() as client:
        return await adapter(entity).fetch(client, since)


@respx.mock
async def test_maps_model_fields():
    items = await run([record("org/model-a")])
    assert len(items) == 1
    item = items[0]
    assert item.external_id == "models:org/model-a"
    assert item.url == "https://huggingface.co/org/model-a"
    assert item.title == "org/model-a (model)"
    assert "10,000 downloads" in item.summary
    assert item.raw["downloads"] == 10_000


@respx.mock
async def test_datasets_get_the_dataset_url_path():
    items = await run([record("org/ds")], entity="datasets")
    assert items[0].url == "https://huggingface.co/datasets/org/ds"
    assert items[0].external_id == "datasets:org/ds"
    assert "(dataset)" in items[0].title


@respx.mock
async def test_noise_tags_are_stripped_from_topics():
    """Hub tags mix licences, arxiv ids and real task tags. Only the useful ones
    should reach the ranker."""
    items = await run([record("org/m")])
    topics = items[0].topics
    assert "llama" in topics and "conversational" in topics
    assert not any(t.startswith(("license:", "arxiv:")) for t in topics)


@respx.mock
async def test_private_repos_are_skipped():
    items = await run([record("org/secret", private=True), record("org/public")])
    assert [i.raw["repo_id"] for i in items] == ["org/public"]


@respx.mock
async def test_stale_repos_are_dropped():
    items = await run([record("org/old", lastModified="2019-01-01T00:00:00Z"), record("org/new")])
    assert [i.raw["repo_id"] for i in items] == ["org/new"]


@respx.mock
async def test_unparseable_timestamp_drops_the_record_rather_than_raising():
    items = await run([record("org/bad", lastModified="not-a-date"), record("org/good")])
    assert [i.raw["repo_id"] for i in items] == ["org/good"]


@respx.mock
async def test_non_list_payload_raises():
    respx.get(MODELS_URL).mock(return_value=httpx.Response(200, json={"error": "nope"}))
    async with httpx.AsyncClient() as client:
        with pytest.raises(ValueError, match="expected a list"):
            await adapter().fetch(client, datetime.now(timezone.utc) - timedelta(days=1))


class TestEngagement:
    """Log scaling exists so a promising newcomer is not rounded to zero next to a
    foundation model with a million downloads."""

    @respx.mock
    async def test_new_model_with_modest_downloads_is_clearly_above_zero(self):
        items = await run([record("org/new", downloads=3_000, likes=5)])
        assert 0.4 < items[0].engagement < 0.85

    @respx.mock
    async def test_more_downloads_always_scores_higher(self):
        items = await run(
            [
                record("org/small", downloads=1_000, likes=0),
                record("org/big", downloads=900_000, likes=0),
            ]
        )
        scores = {i.raw["repo_id"]: i.engagement for i in items}
        assert scores["org/big"] > scores["org/small"]

    @respx.mock
    async def test_engagement_never_exceeds_one(self):
        items = await run([record("org/huge", downloads=50_000_000, likes=90_000)])
        assert items[0].engagement == 1.0

    @respx.mock
    async def test_zero_signal_is_zero(self):
        items = await run([record("org/dead", downloads=0, likes=0)])
        assert items[0].engagement == 0.0


@respx.mock
async def test_limit_is_capped():
    route = respx.get(MODELS_URL).mock(return_value=httpx.Response(200, json=[]))
    a = HuggingFaceAdapter(
        SourceConfig(slug="hf", name="HF", endpoint=MODELS_URL, config={"limit": 99_999})
    )
    async with httpx.AsyncClient() as client:
        await a.fetch(client, datetime.now(timezone.utc) - timedelta(days=1))
    assert route.calls[0].request.url.params["limit"] == "200"
