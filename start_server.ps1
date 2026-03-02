# PowerShell script to start the SehatMind chatbot server
# This script loads the GROQ_API_KEY from system environment variables
# and starts the FastAPI server

Write-Host "Starting SehatMind Chatbot Server..." -ForegroundColor Green
Write-Host ""

# Get the API key from system environment variables
$apiKey = [System.Environment]::GetEnvironmentVariable("GROQ_API_KEY", "User")

# Set it in the current PowerShell session
if ($apiKey) {
    $env:GROQ_API_KEY = $apiKey
    Write-Host "✓ API Key loaded from system environment variables" -ForegroundColor Green
} else {
    Write-Host "⚠ Warning: GROQ_API_KEY not found in system environment variables" -ForegroundColor Yellow
    Write-Host "  The server will run but LLM responses will fail." -ForegroundColor Yellow
}

Write-Host ""

# Change to the script directory
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir

# Start the server
Write-Host "Starting FastAPI server..." -ForegroundColor Cyan
Write-Host ""

python -m uvicorn api.server:app --reload

