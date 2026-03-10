import pytest
from unittest.mock import patch, AsyncMock
from multiroute.providers.google_genai import Client

def test_google_genai_sync_fallback():
    client = Client(api_key="test")
    with patch('google.genai.models.Models.generate_content') as mock_create:
        mock_create.side_effect = [Exception("error1"), "success"]
        
        result = client.models.generate_content(models=["gemini-1.5-pro", "gemini-1.5-flash"], contents="hi")
        
        assert result == "success"
        assert mock_create.call_count == 2
        mock_create.assert_any_call(model="gemini-1.5-pro", contents="hi")
        mock_create.assert_any_call(model="gemini-1.5-flash", contents="hi")

def test_google_genai_sync_all_fail():
    client = Client(api_key="test")
    with patch('google.genai.models.Models.generate_content') as mock_create:
        mock_create.side_effect = [Exception("error1"), Exception("error2")]
        
        with pytest.raises(Exception, match="error2"):
            client.models.generate_content(models=["gemini-1.5-pro", "gemini-1.5-flash"], contents="hi")

@pytest.mark.asyncio
async def test_google_genai_async_fallback():
    client = Client(api_key="test")
    with patch('google.genai.models.AsyncModels.generate_content', new_callable=AsyncMock) as mock_create:
        mock_create.side_effect = [Exception("error1"), "success"]
        
        result = await client.aio.models.generate_content(models=["gemini-1.5-pro", "gemini-1.5-flash"], contents="hi")
        
        assert result == "success"
        assert mock_create.call_count == 2
        mock_create.assert_any_call(model="gemini-1.5-pro", contents="hi")
        mock_create.assert_any_call(model="gemini-1.5-flash", contents="hi")
