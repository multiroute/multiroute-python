import pytest
from unittest.mock import patch, AsyncMock
from multiroute.providers.anthropic import Anthropic, AsyncAnthropic

def test_anthropic_sync_fallback():
    client = Anthropic(api_key="test")
    with patch('anthropic.resources.messages.Messages.create') as mock_create:
        mock_create.side_effect = [Exception("error1"), "success"]
        
        result = client.messages.create(models=["claude-3-opus", "claude-3-sonnet"], max_tokens=100, messages=[])
        
        assert result == "success"
        assert mock_create.call_count == 2
        mock_create.assert_any_call(model="claude-3-opus", max_tokens=100, messages=[])
        mock_create.assert_any_call(model="claude-3-sonnet", max_tokens=100, messages=[])

def test_anthropic_sync_all_fail():
    client = Anthropic(api_key="test")
    with patch('anthropic.resources.messages.Messages.create') as mock_create:
        mock_create.side_effect = [Exception("error1"), Exception("error2")]
        
        with pytest.raises(Exception, match="error2"):
            client.messages.create(models=["claude-3-opus", "claude-3-sonnet"], max_tokens=100, messages=[])

@pytest.mark.asyncio
async def test_anthropic_async_fallback():
    client = AsyncAnthropic(api_key="test")
    with patch('anthropic.resources.messages.AsyncMessages.create', new_callable=AsyncMock) as mock_create:
        mock_create.side_effect = [Exception("error1"), "success"]
        
        result = await client.messages.create(models=["claude-3-opus", "claude-3-sonnet"], max_tokens=100, messages=[])
        
        assert result == "success"
        assert mock_create.call_count == 2
        mock_create.assert_any_call(model="claude-3-opus", max_tokens=100, messages=[])
        mock_create.assert_any_call(model="claude-3-sonnet", max_tokens=100, messages=[])
