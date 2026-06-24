#!/bin/bash
# SkyTrackVision - AirSim Settings Sync Script (Linux/macOS)

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_FILE="$PROJECT_ROOT/settings.json"
AIRSIM_DIR="$HOME/Documents/AirSim"
DEST_FILE="$AIRSIM_DIR/settings.json"

echo "🔄 SkyTrackVision - Settings Sync"
echo ""

# AirSim dizinini oluştur (yoksa)
if [ ! -d "$AIRSIM_DIR" ]; then
    echo "📁 Creating AirSim directory: $AIRSIM_DIR"
    mkdir -p "$AIRSIM_DIR"
fi

# Source dosya kontrolü
if [ ! -f "$SOURCE_FILE" ]; then
    echo "❌ Source file not found: $SOURCE_FILE"
    exit 1
fi

# Backup oluştur
if [ -f "$DEST_FILE" ]; then
    BACKUP_FILE="$DEST_FILE.backup_$(date +%Y%m%d_%H%M%S)"
    echo "💾 Backing up existing settings: $(basename $BACKUP_FILE)"
    cp "$DEST_FILE" "$BACKUP_FILE"
fi

# Kopyala
echo "📋 Copying settings.json..."
cp "$SOURCE_FILE" "$DEST_FILE"

# Doğrula
if [ -f "$DEST_FILE" ]; then
    echo "✅ Settings synced successfully!"
    echo ""
    echo "Source:      $SOURCE_FILE"
    echo "Destination: $DEST_FILE"
    echo ""
    echo "⚠️  You may need to restart AirSim!"
else
    echo "❌ Failed to sync settings!"
    exit 1
fi
