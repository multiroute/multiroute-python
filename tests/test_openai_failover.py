import pytest
from unittest.mock import patch, AsyncMock
from multiroute.providers.openai import OpenAI, AsyncOpenAI

def test_sync_single_model_success():
    client = OpenAI(api_key="test")
    with patch('openai.resources.chat.Completions.create') as mock_create:
        mock_create.return_value = "success"
        
        result = client.chat.completions.create(model="gpt-3.5-turbo", messages=[{"role": "user", "content": "hi"}])
        
        assert result == "success"
        mock_create.assert_called_once_with(model="gpt-3.5-turbo", messages=[{"role": "user", "content": "hi"}])

def test_sync_failover_success_on_second():
    client = OpenAI(api_key="test")
    with patch('openai.resources.chat.Completions.create') as mock_create:
        mock_create.side_effect = [Exception("error1"), "success"]
        
        result = client.chat.completions.create(models=["gpt-4", "gpt-3.5-turbo"], messages=[])
        
        assert result == "success"
        assert mock_create.call_count == 2
        mock_create.assert_any_call(model="gpt-4", messages=[])
        mock_create.assert_any_call(model="gpt-3.5-turbo", messages=[])

def test_sync_failover_all_fail():
    client = OpenAI(api_key="test")
    with patch('openai.resources.chat.Completions.create') as mock_create:
        mock_create.side_effect = [Exception("error1"), Exception("error2")]
        
        with pytest.raises(Exception, match="error2"):
            client.chat.completions.create(models=["gpt-4", "gpt-3.5-turbo"], messages=[])

@pytest.mark.asyncio
async def test_async_failover_success_on_second():
    client = AsyncOpenAI(api_key="test")
    with patch('openai.resources.chat.AsyncCompletions.create', new_callable=AsyncMock) as mock_create:
        mock_create.side_effect = [Exception("error1"), "success"]
        
        result = await client.chat.completions.create(models=["gpt-4", "gpt-3.5-turbo"], messages=[])
        
        assert result == "success"
        assert mock_create.call_count == 2
        mock_create.assert_any_call(model="gpt-4", messages=[])
        mock_create.assert_any_call(model="gpt-3.5-turbo", messages=[])

@pytest.mark.asyncio
async def test_async_failover_all_fail():
    client = AsyncOpenAI(api_key="test")
    with patch('openai.resources.chat.AsyncCompletions.create', new_callable=AsyncMock) as mock_create:
        mock_create.side_effect = [Exception("error1"), Exception("error2")]
        
        with pytest.raises(Exception, match="error2"):
            await client.chat.completions.create(models=["gpt-4", "gpt-3.5-turbo"], messages=[])

def test_responses_sync_failover():
    client = OpenAI(api_key="test")
    with patch('openai.resources.responses.Responses.create') as mock_create:
        mock_create.side_effect = [Exception("error1"), "success"]
        
        result = client.responses.create(models=["gpt-4", "gpt-3.5-turbo"], input="hello")
        
        assert result == "success"
        assert mock_create.call_count == 2
        mock_create.assert_any_call(model="gpt-4", input="hello")
        mock_create.assert_any_call(model="gpt-3.5-turbo", input="hello")

@pytest.mark.asyncio
async def test_responses_async_failover():
    client = AsyncOpenAI(api_key="test")
    with patch('openai.resources.responses.AsyncResponses.create', new_callable=AsyncMock) as mock_create:
        mock_create.side_effect = [Exception("error1"), "success"]
        
        result = await client.responses.create(models=["gpt-4", "gpt-3.5-turbo"], input="hello")
        
        assert result == "success"
        assert mock_create.call_count == 2
        mock_create.assert_any_call(model="gpt-4", input="hello")
        mock_create.assert_any_call(model="gpt-3.5-turbo", input="hello")
