"""
API客户端测试
"""
import pytest
import asyncio
import aiohttp
from unittest.mock import AsyncMock, patch
from src.api_clients import MockAPI, OpenAIAPI, ClaudeAPI, OllamaAPI, create_api_client

@pytest.mark.asyncio
async def test_mock_api():
    """测试Mock API功能"""
    config = {"model": "test-model"}
    api = MockAPI(config)
    
    # 测试不同类别的响应
    test_cases = [
        ("A child has fever", "healthcare"),
        ("Student asks about tradition", "education"),
        ("Someone offers money for lies", "ethics"),
        ("Flood evacuation needed", "emergency")
    ]
    
    for prompt, expected_category in test_cases:
        response = await api.generate_response(prompt, language="en")
        assert isinstance(response, str)
        assert len(response) > 0

@pytest.mark.asyncio
async def test_openai_api_mock():
    """测试OpenAI API（使用模拟响应）"""
    config = {
        "api_key": "test-key",
        "model": "gpt-3.5-turbo",
        "base_url": "https://api.openai.com/v1"
    }
    
    api = OpenAIAPI(config)
    
    # 模拟成功响应
    mock_response = {
        "choices": [
            {
                "message": {
                    "content": "This is a test response"
                }
            }
        ]
    }
    
    with patch('aiohttp.ClientSession.post') as mock_post:
        mock_post.return_value.__aenter__.return_value.status = 200
        mock_post.return_value.__aenter__.return_value.json = AsyncMock(return_value=mock_response)
        
        response = await api.generate_response("Test prompt")
        assert response == "This is a test response"

@pytest.mark.asyncio
async def test_claude_api_mock():
    """测试Claude API（使用模拟响应）"""
    config = {
        "api_key": "test-key",
        "model": "claude-3-sonnet-20240229",
        "base_url": "https://api.anthropic.com"
    }
    
    api = ClaudeAPI(config)
    
    # 模拟成功响应
    mock_response = {
        "content": [
            {
                "text": "This is a Claude response"
            }
        ]
    }
    
    with patch('aiohttp.ClientSession.post') as mock_post:
        mock_post.return_value.__aenter__.return_value.status = 200
        mock_post.return_value.__aenter__.return_value.json = AsyncMock(return_value=mock_response)
        
        response = await api.generate_response("Test prompt")
        assert response == "This is a Claude response"

@pytest.mark.asyncio
async def test_ollama_api_mock():
    """测试Ollama API（使用模拟响应）"""
    config = {
        "base_url": "http://localhost:11434",
        "model": "llama2"
    }
    
    api = OllamaAPI(config)
    
    # 模拟成功响应
    mock_response = {
        "response": "This is an Ollama response"
    }
    
    with patch('aiohttp.ClientSession.post') as mock_post:
        mock_post.return_value.__aenter__.return_value.status = 200
        mock_post.return_value.__aenter__.return_value.json = AsyncMock(return_value=mock_response)
        
        response = await api.generate_response("Test prompt")
        assert response == "This is an Ollama response"

def test_create_api_client():
    """测试API客户端工厂方法"""
    config = {
        "openai": {
            "api_key": "test-key",
            "model": "gpt-3.5-turbo",
            "base_url": "https://api.openai.com/v1"
        },
        "claude": {
            "api_key": "test-key",
            "model": "claude-3-sonnet-20240229",
            "base_url": "https://api.anthropic.com"
        },
        "ollama": {
            "base_url": "http://localhost:11434",
            "model": "llama2"
        }
    }
    
    # 测试不同模型的客户端创建
    openai_client = create_api_client("openai", config)
    assert isinstance(openai_client, OpenAIAPI)
    
    claude_client = create_api_client("claude", config)
    assert isinstance(claude_client, ClaudeAPI)
    
    ollama_client = create_api_client("ollama", config)
    assert isinstance(ollama_client, OllamaAPI)
    
    # 测试未知模型（应该返回Mock API）
    unknown_client = create_api_client("unknown", config)
    assert isinstance(unknown_client, MockAPI)

@pytest.mark.asyncio
async def test_api_error_handling():
    """测试API错误处理"""
    config = {
        "api_key": "invalid-key",
        "model": "test-model",
        "base_url": "https://api.example.com"
    }
    
    api = OpenAIAPI(config)
    
    # 模拟错误响应
    with patch('aiohttp.ClientSession.post') as mock_post:
        mock_post.return_value.__aenter__.return_value.status = 401
        mock_post.return_value.__aenter__.return_value.text = AsyncMock(return_value="Unauthorized")
        
        with pytest.raises(Exception):
            await api.generate_response("Test prompt")

@pytest.mark.asyncio
async def test_api_timeout_handling():
    """测试API超时处理"""
    config = {
        "api_key": "test-key",
        "model": "test-model",
        "base_url": "https://api.example.com"
    }
    
    api = OpenAIAPI(config)
    
    # 模拟超时
    with patch('aiohttp.ClientSession.post') as mock_post:
        mock_post.side_effect = asyncio.TimeoutError()
        
        with pytest.raises(Exception):
            await api.generate_response("Test prompt")

if __name__ == "__main__":
    pytest.main([__file__, "-v"])