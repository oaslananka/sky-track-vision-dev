# SkyTrackVision - AirSim Settings Sync Script
# Bu script settings.json'ı AirSim dizinine kopyalar

$ProjectRoot = $PSScriptRoot
$SourceFile = Join-Path $ProjectRoot "settings.json"
$AirSimDir = Join-Path $env:USERPROFILE "Documents\AirSim"
$DestFile = Join-Path $AirSimDir "settings.json"

Write-Host "🔄 SkyTrackVision - Settings Sync" -ForegroundColor Cyan
Write-Host ""

# AirSim dizinini oluştur (yoksa)
if (-not (Test-Path $AirSimDir)) {
    Write-Host "📁 Creating AirSim directory: $AirSimDir" -ForegroundColor Yellow
    New-Item -ItemType Directory -Path $AirSimDir -Force | Out-Null
}

# Source dosya kontrolü
if (-not (Test-Path $SourceFile)) {
    Write-Host "❌ Source file not found: $SourceFile" -ForegroundColor Red
    exit 1
}

# Backup oluştur
if (Test-Path $DestFile) {
    $BackupFile = "$DestFile.backup_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
    Write-Host "💾 Backing up existing settings: $(Split-Path $BackupFile -Leaf)" -ForegroundColor Yellow
    Copy-Item $DestFile $BackupFile
}

# Kopyala
Write-Host "📋 Copying settings.json..." -ForegroundColor Green
Copy-Item $SourceFile $DestFile -Force

# Doğrula
if (Test-Path $DestFile) {
    Write-Host "✅ Settings synced successfully!" -ForegroundColor Green
    Write-Host ""
    Write-Host "Source:      $SourceFile" -ForegroundColor Gray
    Write-Host "Destination: $DestFile" -ForegroundColor Gray
    Write-Host ""
    Write-Host "⚠️  AirSim'i yeniden başlatmanız gerekebilir!" -ForegroundColor Yellow
} else {
    Write-Host "❌ Failed to sync settings!" -ForegroundColor Red
    exit 1
}
