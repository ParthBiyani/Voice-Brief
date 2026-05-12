from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from voicebrief.sources.base import RawItem, SourceAdapter, SourceConfig, _Registry


def make_item(**over) -> RawItem:
    base = dict(
        external_id="abc",
        url="https://example.com/a?utm_source=x",
        title="  A   Title\nwith  whitespace ",
        published_at=datetime(2026, 6, 1, 12, 0),
    )
    return RawItem(**{**base, **over})


class TestRawItem:
    def test_collapses_whitespace_in_title(self):
        assert make_item().title == "A Title with whitespace"

    def test_naive_datetime_is_assumed_utc(self):
        assert make_item().published_at.tzinfo == timezone.utc

    def test_aware_datetime_is_preserved(self):
        tz = timezone.utc
        item = make_item(published_at=datetime(2026, 6, 1, 12, 0, tzinfo=tz))
        assert item.published_at.tzinfo == tz

    @pytest.mark.parametrize(("given", "expected"), [(-1.0, 0.0), (0.5, 0.5), (7.0, 1.0)])
    def test_engagement_is_clamped_to_unit_range(self, given, expected):
        assert make_item(engagement=given).engagement == expected

    def test_fingerprint_ignores_query_string_and_case(self):
        a = make_item(url="https://example.com/a?utm_source=twitter", title="Same Title")
        b = make_item(url="https://example.com/a/", title="same title")
        assert a.fingerprint() == b.fingerprint()

    def test_fingerprint_differs_on_different_content(self):
        assert make_item(title="One").fingerprint() != make_item(title="Two").fingerprint()

    def test_title_is_required(self):
        with pytest.raises(ValidationError):
            RawItem(external_id="x", url="https://e.com", published_at=datetime.now())


class TestRegistry:
    def test_build_returns_registered_adapter(self):
        reg = _Registry()

        @reg.register
        class Dummy(SourceAdapter):
            kind = "dummy"

            async def fetch(self, client, since):
                return []

        cfg = SourceConfig(slug="s", name="S", endpoint="https://e.com")
        assert isinstance(reg.build("dummy", cfg), Dummy)

    def test_unknown_kind_raises_with_helpful_message(self):
        reg = _Registry()
        cfg = SourceConfig(slug="s", name="S", endpoint="https://e.com")
        with pytest.raises(LookupError, match="No adapter registered"):
            reg.build("nope", cfg)

    def test_adapter_without_kind_is_rejected(self):
        reg = _Registry()
        with pytest.raises(ValueError, match="must define a `kind`"):

            @reg.register
            class NoKind(SourceAdapter):
                async def fetch(self, client, since):
                    return []
