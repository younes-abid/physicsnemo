#!/bin/bash

# Multi-Scale Cross-Attention Enhancement - Version 1 Application Script
# This script copies the enhanced v1 files to their original locations in the physicsnemo codebase

set -e  # Exit on any error

echo "🚀 Applying Multi-Scale Cross-Attention Enhancement v1..."
echo "=================================================="

# Get the script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "📁 Script directory: $SCRIPT_DIR"

# Define source and destination paths
V1_DIR="$SCRIPT_DIR"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../../../../../.." && pwd)"

echo "📁 Repository root: $REPO_ROOT"
echo ""

# Backup original files first
echo "💾 Creating backups of original files..."
BACKUP_DIR="$REPO_ROOT/outputs/backups/cross_attention_v1_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"

# Copy original files to backup
cp "$REPO_ROOT/physicsnemo/models/diffusion/preconditioning.py" "$BACKUP_DIR/"
cp "$REPO_ROOT/physicsnemo/models/diffusion/layers.py" "$BACKUP_DIR/"
cp "$REPO_ROOT/physicsnemo/models/diffusion/song_unet.py" "$BACKUP_DIR/"

echo "✅ Backups created in: $BACKUP_DIR"
echo ""

# Apply the enhanced files
echo "🔄 Applying enhanced files..."

# Copy preconditioning.py
echo "   📝 Copying enhanced preconditioning.py..."
cp "$V1_DIR/preconditioning.py" "$REPO_ROOT/physicsnemo/models/diffusion/preconditioning.py"

# Copy layers.py
echo "   📝 Copying enhanced layers.py..."
cp "$V1_DIR/layers.py" "$REPO_ROOT/physicsnemo/models/diffusion/layers.py"

# Copy song_unet.py
echo "   📝 Copying enhanced song_unet.py..."
cp "$V1_DIR/song_unet.py" "$REPO_ROOT/physicsnemo/models/diffusion/song_unet.py"

echo ""
echo "✅ All enhanced files applied successfully!"
echo ""

# Verify the files exist
echo "🔍 Verifying applied files..."
for file in preconditioning.py layers.py song_unet.py; do
    if [ -f "$REPO_ROOT/physicsnemo/models/diffusion/$file" ]; then
        echo "   ✅ $file - OK"
    else
        echo "   ❌ $file - MISSING"
        exit 1
    fi
done

echo ""
echo "🎉 Multi-Scale Cross-Attention Enhancement v1 applied successfully!"
echo ""
echo "📋 Summary of changes:"
echo "   • Cross-attention support added to UNetBlock"
echo "   • Multi-scale LR feature processing in SongUNet"
echo "   • Weather-aware attention patterns"
echo "   • Backward compatibility maintained"
echo "   • Optional lr_features parameter added"
echo ""
echo "🔗 Usage example:"
echo "   model = SongUNet(use_cross_attention=True, weather_aware=True)"
echo "   output = model(x, noise_labels, class_labels, lr_features=lr_data)"
echo ""
echo "📖 For detailed documentation, see: $V1_DIR/README.md"
echo ""
echo "💡 To revert changes, restore files from: $BACKUP_DIR"
echo "=================================================="