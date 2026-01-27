"""
RAM+ Image Tagger Server
Provides HTTP API compatible with AI-image-auto-tagger
Single model: RAM+ (Recognize-Anything-Plus)
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager
import uvicorn
import argparse
import os
from typing import Optional
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from pillow_heif import register_heif_opener
from ram import get_transform, inference_ram as inference
from ram.models import ram_plus
from pathlib import Path
import tempfile

from xmphandler import XMPHandler

# Configuration
MODEL_REPO_ID = "xinyu1205/recognize-anything-plus-model"
MODEL_FILENAME = "ram_plus_swin_large_14m.pth"
IMAGE_SIZE = 384
DEFAULT_PREFIX = "ram"
MIN_FILE_SIZE_BYTES = 1024

# Global tagger
tagger = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, cleanup on shutdown"""
    global tagger
    
    print("🚀 RAM+ Image Tagger API Server starting...")
    
    # Download and load model
    try:
        print("  Downloading RAM+ model from HuggingFace...")
        model_path = hf_hub_download(
            repo_id=MODEL_REPO_ID,
            filename=MODEL_FILENAME
        )
        
        print("  Loading RAM+ model...")
        register_heif_opener()
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        transform = get_transform(image_size=IMAGE_SIZE)
        model = ram_plus(pretrained=model_path, image_size=IMAGE_SIZE, vit='swin_l')
        model.eval()
        model = model.to(device)
        
        tagger = {
            'model': model,
            'transform': transform,
            'device': device
        }
        
        print(f"  [OK] RAM+ model loaded on {device}")
    except Exception as e:
        print(f"  [FAIL] Failed to load model: {e}")
        import traceback
        traceback.print_exc()
    
    yield
    
    # Cleanup
    print("\n🛑 Shutting down RAM+ API Server")
    if tagger and torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("✅ Cleanup complete")


app = FastAPI(
    title="RAM+ Image Tagger API",
    description="Tag images using RAM+ (Recognize-Anything-Plus) model",
    version="1.0.0",
    lifespan=lifespan
)


# ============================================================================
# Request/Response Models (compatible with AI-image-auto-tagger)
# ============================================================================

class TagRequest(BaseModel):
    image_path: str = Field(..., description="Absolute path to the image file")
    model: Optional[str] = Field(None, description="Model (ignored, RAM+ only)")
    general_thresh: float = Field(0.35, description="Ignored (RAM+ specific)")
    character_thresh: float = Field(0.85, description="Ignored (RAM+ specific)")
    hide_rating_tags: bool = Field(False, description="Ignored (RAM+ has no rating tags)")
    character_tags_first: bool = Field(False, description="Ignored (RAM+ specific)")
    remove_separator: bool = Field(False, description="Remove underscore separator in tags")
    hierarchical_prefix: str = Field(DEFAULT_PREFIX, description="Hierarchical prefix for XMP tags")
    save_xmp: bool = Field(False, description="Save tags to XMP sidecar file")
    overwrite_tags: bool = Field(False, description="Overwrite existing tags")


class TagResponse(BaseModel):
    status: str
    message: str
    model: str = Field(default="ram-plus", description="Model used")
    tags: list[str] = Field(default_factory=list, description="List of detected tags")
    tags_hierarchical: dict[str, list[str]] = Field(
        default_factory=dict,
        description="Tags grouped by category (general only for RAM+)"
    )


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/")
async def root():
    """Health check - compatible with AI-image-auto-tagger"""
    return {
        "status": "running",
        "service": "RAM+ Image Tagger API",
        "version": "1.0.0",
        "model_cache": {"cached_models": ["ram-plus"], "cache_size": 1, "max_size": 1},
        "available_models": ["ram-plus"]
    }


@app.post("/tag", response_model=TagResponse)
async def tag_image(request: TagRequest):
    """
    Tag a single image using RAM+ model
    API compatible with AI-image-auto-tagger
    """
    global tagger
    
    if tagger is None:
        raise HTTPException(status_code=503, detail="RAM+ model not loaded")
    
    # Validate file
    if not os.path.exists(request.image_path):
        raise HTTPException(status_code=404, detail=f"Image file not found: {request.image_path}")
    
    if not os.path.isfile(request.image_path):
        raise HTTPException(status_code=400, detail=f"Path is not a file: {request.image_path}")
    
    # Check file size
    file_size = os.path.getsize(request.image_path)
    if file_size < MIN_FILE_SIZE_BYTES:
        return TagResponse(
            status="skipped",
            message=f"File size too small ({file_size} bytes)",
            model="ram-plus",
            tags=[],
            tags_hierarchical={"general": []}
        )
    
    # Check if already tagged
    if not request.overwrite_tags and request.save_xmp:
        sidecar_files = XMPHandler.get_xmp_sidecars_for_image(request.image_path)
        if sidecar_files:
            try:
                handler = XMPHandler(sidecar_files[0])
                all_subjects = handler.get_all_subjects()
                
                # Extract existing tags
                existing_tags = []
                prefix_with_pipe = f"{request.hierarchical_prefix}|"
                
                for subject in all_subjects:
                    if subject.startswith(prefix_with_pipe):
                        tag = subject[len(prefix_with_pipe):]
                        if tag and tag not in existing_tags:
                            existing_tags.append(tag)
                
                if existing_tags:
                    existing_tags.sort()
                    return TagResponse(
                        status="skipped",
                        message=f"Loaded {len(existing_tags)} tags from existing XMP sidecar",
                        model="ram-plus",
                        tags=existing_tags,
                        tags_hierarchical={"general": existing_tags}
                    )
            except Exception as e:
                print(f"Warning: Could not read XMP: {e}")
    
    try:
        # Load image
        try:
            image = Image.open(request.image_path)
            if image.mode != 'RGB':
                image = image.convert('RGB')
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Could not load image: {e}")
        
        # Run inference
        with torch.no_grad():
            torch_image = tagger['transform'](image).unsqueeze(0).to(tagger['device'])
            tags_str = inference(torch_image, tagger['model'])[0]
        
        if not tags_str:
            raise HTTPException(status_code=400, detail="No tags generated")
        
        # Parse tags
        tags = [tag.strip() for tag in tags_str.replace("|", ",").split(",") if tag.strip()]
        
        # Apply separator removal if requested
        if request.remove_separator:
            tags = [tag.replace("_", " ") for tag in tags]
        
        # Sort alphabetically
        tags.sort()
        
        # Save to XMP if requested
        if request.save_xmp:
            try:
                sidecar_files = XMPHandler.get_xmp_sidecars_for_image(request.image_path)
                if not sidecar_files:
                    sidecar_files = [XMPHandler.create_xmp_sidecar(request.image_path, False)]
                
                for xmp_file in sidecar_files:
                    handler = XMPHandler(xmp_file)
                    for tag in tags:
                        handler.add_hierarchical_subject(f"{request.hierarchical_prefix}|{tag}")
                    handler.save()
            except Exception as e:
                print(f"Warning: Could not save XMP: {e}")
        
        return TagResponse(
            status="success",
            message=f"Successfully tagged image with {len(tags)} tags",
            model="ram-plus",
            tags=tags,
            tags_hierarchical={"general": tags}
        )
    
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Tagging failed: {str(e)}")


@app.post("/tag-upload", response_model=TagResponse)
async def tag_upload(
    file: UploadFile = File(..., description="Image file to tag"),
    model: Optional[str] = Form(None, description="Model ID (ignored, RAM+ only)"),
    general_thresh: float = Form(0.35, ge=0.0, le=1.0),
    character_thresh: float = Form(0.85, ge=0.0, le=1.0),
    hide_rating_tags: bool = Form(False),
    character_tags_first: bool = Form(False),
    remove_separator: bool = Form(False),
    hierarchical_prefix: str = Form(DEFAULT_PREFIX),
    save_xmp: bool = Form(False)
):
    """
    Tag an image by uploading it as multipart form data
    Compatible with AI-image-auto-tagger endpoint
    
    Args:
        file: Image file to upload and tag
        model: Model ID (ignored, uses RAM+ only)
        general_thresh: General tag threshold (ignored)
        character_thresh: Character tag threshold (ignored)
        hide_rating_tags: Hide rating tags (ignored)
        character_tags_first: Place character tags first (ignored)
        remove_separator: Remove underscore separator in tags
        hierarchical_prefix: Hierarchical prefix for XMP tags
        save_xmp: Save tags to XMP sidecar file
        
    Returns:
        TagResponse with processing status and detected tags
    """
    global tagger
    
    if tagger is None:
        raise HTTPException(status_code=503, detail="RAM+ model not loaded")
    
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")
    
    # Check allowed file extensions
    allowed_extensions = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".tiff", ".heif", ".heic"}
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400, 
            detail=f"File type not supported. Allowed: {', '.join(allowed_extensions)}"
        )
    
    try:
        # Create temporary file with original extension
        temp_dir = tempfile.gettempdir()
        temp_path = os.path.join(temp_dir, file.filename)
        
        # Save uploaded file to temp location
        content = await file.read()
        with open(temp_path, "wb") as temp_file:
            temp_file.write(content)
        
        # Load and process image
        try:
            image = Image.open(temp_path)
            if image.mode != 'RGB':
                image = image.convert('RGB')
        except Exception as e:
            os.remove(temp_path)
            raise HTTPException(status_code=400, detail=f"Failed to open image: {str(e)}")
        
        # Run inference
        try:
            with torch.no_grad():
                torch_image = tagger['transform'](image).unsqueeze(0).to(tagger['device'])
                tags_str = inference(torch_image, tagger['model'])[0]
        except Exception as e:
            os.remove(temp_path)
            raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")
        
        # Clean up temp file
        os.remove(temp_path)
        
        if not tags_str:
            return TagResponse(
                status="skipped",
                message="No tags detected",
                model="ram-plus",
                tags=[],
                tags_hierarchical={"general": []}
            )
        
        # Parse tags (handle both | and , separators)
        tags = [tag.strip() for tag in tags_str.replace("|", ",").split(",") if tag.strip()]
        
        # Apply separator removal if requested
        if remove_separator:
            tags = [tag.replace("_", " ") for tag in tags]
        
        # Sort alphabetically
        tags.sort()
        
        return TagResponse(
            status="success",
            message=f"Successfully tagged image with {len(tags)} tags",
            model="ram-plus",
            tags=tags,
            tags_hierarchical={"general": tags}
        )
    
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Upload processing error: {str(e)}")


@app.post("/tag-batch")
async def tag_batch(
    image_path: str,
    hierarchical_prefix: str = DEFAULT_PREFIX,
    overwrite_tags: bool = False
):
    """
    Tag all images in a directory recursively
    Simplified version compatible with AI-image-auto-tagger
    """
    global tagger
    
    if tagger is None:
        raise HTTPException(status_code=503, detail="RAM+ model not loaded")
    
    if not os.path.exists(image_path):
        raise HTTPException(status_code=404, detail=f"Directory not found: {image_path}")
    
    if not os.path.isdir(image_path):
        raise HTTPException(status_code=400, detail=f"Path is not a directory: {image_path}")
    
    stats = {
        "total_files": 0,
        "images_processed": 0,
        "images_skipped": 0,
        "errors": 0,
        "tags_generated": 0
    }
    
    try:
        for root, dirs, files in os.walk(image_path):
            for fname in sorted(files):
                file_path = os.path.join(root, fname)
                
                # Skip XMP files
                if fname.lower().endswith('.xmp'):
                    continue
                
                stats["total_files"] += 1
                
                try:
                    # Try to load image
                    try:
                        image = Image.open(file_path)
                        if image.mode != 'RGB':
                            image = image.convert('RGB')
                    except:
                        stats["images_skipped"] += 1
                        continue
                    
                    # Run inference
                    with torch.no_grad():
                        torch_image = tagger['transform'](image).unsqueeze(0).to(tagger['device'])
                        tags_str = inference(torch_image, tagger['model'])[0]
                    
                    if not tags_str:
                        stats["images_skipped"] += 1
                        continue
                    
                    # Parse and save
                    tags = [tag.strip() for tag in tags_str.replace("|", ",").split(",") if tag.strip()]
                    tags.sort()
                    
                    # Save XMP
                    sidecar_files = XMPHandler.get_xmp_sidecars_for_image(file_path)
                    if not sidecar_files:
                        sidecar_files = [XMPHandler.create_xmp_sidecar(file_path, False)]
                    
                    for xmp_file in sidecar_files:
                        handler = XMPHandler(xmp_file)
                        for tag in tags:
                            handler.add_hierarchical_subject(f"{hierarchical_prefix}|{tag}")
                        handler.save()
                    
                    stats["images_processed"] += 1
                    stats["tags_generated"] += len(tags)
                    print(f"✅ {fname}: {len(tags)} tags")
                    
                except Exception as e:
                    stats["errors"] += 1
                    print(f"❌ {fname}: {e}")
        
        return {
            "status": "success",
            "message": "Batch tagging complete",
            "directory": image_path,
            "model": "ram-plus",
            "stats": stats
        }
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Batch processing failed: {str(e)}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="RAM+ Image Tagger API Server"
    )
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("-p", "--port", type=int, default=8765, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    
    args = parser.parse_args()
    
    print(f"Starting RAM+ API on {args.host}:{args.port}")
    print(f"API docs: http://{args.host}:{args.port}/docs")
    
    uvicorn.run(
        "server:app",
        host=args.host,
        port=args.port,
        reload=args.reload
    )


if __name__ == "__main__":
    main()
