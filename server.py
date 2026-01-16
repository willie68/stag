#!/usr/bin/env python3

"""
FastAPI REST Server for STAG (Stephan's Automatic Image Tagger)
Provides HTTP endpoints to tag images with recognize-anything (RAM) model via SKTagger
"""

from stag import SKTagger
from config import MODEL_REPO_ID, MODEL_FILENAME, DEFAULT_PREFIX, IMAGE_SIZE
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager
import uvicorn
import argparse
import os
from typing import Optional
import torch
from huggingface_hub import hf_hub_download

# Tagger instance loaded on startup
tagger: Optional[SKTagger] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler - loads models on startup and cleans up on shutdown"""
    global tagger
    
    # Startup
    print("Loading AI models...")
    
    # Initialize STAG tagger
    print("  Loading RAM (recognize-anything) model via SKTagger...")
    try:
        # Download the model file (same as GUI)
        print("  Downloading model from HuggingFace...")
        pretrained = hf_hub_download(
            repo_id=MODEL_REPO_ID,
            filename=MODEL_FILENAME
        )
        print(f"  Model downloaded to: {pretrained}")
        
        # Initialize SKTagger with the actual model path (same as GUI)
        tagger = SKTagger(
            model_path=pretrained,  # Use actual path, not None!
            image_size=IMAGE_SIZE,
            force_tagging=False,
            test_mode=False,
            prefer_exact_filenames=False,
            tag_prefix=DEFAULT_PREFIX
        )
        print(f"  [OK] RAM model loaded (device: {tagger.device})")
    except Exception as e:
        print(f"  [FAIL] Failed to load RAM model: {e}")
        import traceback
        traceback.print_exc()
    
    print("Model loading complete")
    
    yield
    
    # Shutdown (cleanup if needed)
    print("Shutting down STAG API")
    if tagger is not None:
        # Clean up CUDA memory if using GPU
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

app = FastAPI(
    title="STAG API",
    description="Tag images using recognize-anything (RAM) model",
    version="1.0.0",
    lifespan=lifespan
)

# Request model with defaults from stag.py
class TagRequest(BaseModel):
    image_path: str = Field(..., description="Absolute path to the image file")
    tag_prefix: str = Field(DEFAULT_PREFIX, description="Hierarchical prefix for XMP tags (e.g., 'st', 'ram')")
    force_tagging: bool = Field(False, description="Force tagging even if already tagged")
    test_mode: bool = Field(False, description="Test mode - don't write to disk")
    prefer_exact_filenames: bool = Field(False, description="Use darktable-compatible filenames for XMP sidecars")
    save_xmp: bool = Field(True, description="Save tags to XMP sidecar file")

class TagResponse(BaseModel):
    status: str
    message: str
    tags: list[str] = Field(default_factory=list, description="List of detected tags")
    device: str = Field("", description="Device used for inference")

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "running",
        "service": "STAG API",
        "version": "1.0.0",
        "device": str(tagger.device) if tagger else "not loaded",
        "models": {
            "ram": tagger is not None
        }
    }

@app.post("/tag", response_model=TagResponse)
async def tag_image(request: TagRequest):
    """
    Tag a single image using SKTagger and optionally save to XMP sidecar
    
    Args:
        request: TagRequest with image path and tagging parameters
        
    Returns:
        TagResponse with processing status and detected tags
    """
    # Check if tagger is loaded
    if tagger is None:
        raise HTTPException(status_code=503, detail="RAM model not loaded")
    
    # Validate file exists
    if not os.path.exists(request.image_path):
        raise HTTPException(status_code=404, detail=f"Image file not found: {request.image_path}")
    
    if not os.path.isfile(request.image_path):
        raise HTTPException(status_code=400, detail=f"Path is not a file: {request.image_path}")
    
    # Check file size
    file_size = os.path.getsize(request.image_path)
    if file_size < 1000:  # Minimum reasonable file size
        return TagResponse(
            status="skipped",
            message=f"File size too small ({file_size} bytes)",
            tags=[],
            device=str(tagger.device)
        )
    
    # Check if XMP already exists when force_tagging=False
    if not request.force_tagging and request.save_xmp:
        from xmphandler import XMPHandler
        sidecar_files = XMPHandler.get_xmp_sidecars_for_image(request.image_path)
        if len(sidecar_files) > 0 and tagger.is_already_tagged(sidecar_files):
            # Read existing tags from XMP sidecar
            try:
                handler = XMPHandler(sidecar_files[0])
                all_subjects = handler.get_all_subjects()
                
                # Filter tags by hierarchical prefix (e.g. "st|tag" -> "tag")
                existing_tags = []
                prefix_with_pipe = f"{request.tag_prefix}|"
                
                for subject in all_subjects:
                    # If subject has the prefix, extract the tag part
                    if subject.startswith(prefix_with_pipe):
                        tag = subject[len(prefix_with_pipe):]
                        if tag not in existing_tags:
                            existing_tags.append(tag)
                    # Also include subjects without prefix (legacy)
                    elif request.tag_prefix not in subject and subject not in existing_tags:
                        existing_tags.append(subject)
                
                return TagResponse(
                    status="skipped",
                    message=f"Loaded {len(existing_tags)} tags from existing XMP sidecar (not processed with AI)",
                    tags=existing_tags,
                    device=str(tagger.device)
                )
            except Exception as e:
                # If reading fails, fall back to processing the image
                print(f"Warning: Could not read existing XMP: {e}")
    
    try:
        # Load and process image using SKTagger
        pil_image, loader = tagger.load_image(request.image_path)
        
        if pil_image is None:
            raise HTTPException(status_code=400, detail=f"Could not load image (loader: {loader})")
        
        # Get tags using SKTagger
        tags_str = tagger.get_tags_for_image(pil_image)
        
        if not tags_str:
            raise HTTPException(status_code=400, detail="No tags generated")
        
        # Convert comma/pipe separated tags to list
        final_tags = [tag.strip() for tag in tags_str.replace("|", ",").split(",") if tag.strip()]
        
        # Save to XMP if requested
        if request.save_xmp and not request.test_mode:
            try:
                from xmphandler import XMPHandler
                
                sidecar_files = XMPHandler.get_xmp_sidecars_for_image(request.image_path)
                tagger.save_tags(request.image_path, sidecar_files, final_tags)
            except Exception as e:
                print(f"Warning: Could not save XMP: {e}")
        
        return TagResponse(
            status="success",
            message=f"Successfully tagged image with {len(final_tags)} tags (loader: {loader})",
            tags=final_tags,
            device=str(tagger.device)
        )
    
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Tagging failed: {str(e)}")

@app.post("/tag-batch")
async def tag_batch(directory: str, tag_prefix: str = DEFAULT_PREFIX, force_tagging: bool = False, test_mode: bool = False):
    """
    Tag all images in a directory recursively using SKTagger
    
    Args:
        directory: Directory to scan for images
        tag_prefix: Hierarchical prefix for XMP tags
        force_tagging: Force tagging even if already tagged
        test_mode: Test mode - don't write to disk
        
    Returns:
        Summary of processing results
    """
    if tagger is None:
        raise HTTPException(status_code=503, detail="RAM model not loaded")
    
    if not os.path.exists(directory):
        raise HTTPException(status_code=404, detail=f"Directory not found: {directory}")
    
    if not os.path.isdir(directory):
        raise HTTPException(status_code=400, detail=f"Path is not a directory: {directory}")
    
    try:
        from xmphandler import XMPHandler
        import threading
        
        # Update tagger configuration
        tagger.force_tagging = force_tagging
        tagger.test_mode = test_mode
        tagger.tag_prefix = tag_prefix
        
        # Process directory
        stats = {
            "total_images": 0,
            "tagged": 0,
            "skipped": 0,
            "errors": 0
        }
        
        stop_event = threading.Event()
        
        for current_dir, _, file_list in os.walk(directory):
            for fname in sorted(file_list):
                if stop_event.is_set():
                    break
                
                file_path = os.path.join(current_dir, fname)
                file_ext = os.path.splitext(fname)[1].lower()
                
                # Skip XMP and known non-image files
                if file_ext == ".xmp":
                    continue
                
                # Load and tag image
                try:
                    pil_image, loader = tagger.load_image(file_path)
                    
                    if pil_image is None:
                        stats["skipped"] += 1
                        continue
                    
                    stats["total_images"] += 1
                    
                    # Check if already tagged
                    sidecar_files = XMPHandler.get_xmp_sidecars_for_image(file_path)
                    if tagger.is_already_tagged(sidecar_files):
                        stats["skipped"] += 1
                        print(f"Skipped (already tagged): {fname}")
                        continue
                    
                    # Get tags
                    tags_str = tagger.get_tags_for_image(pil_image)
                    if tags_str:
                        tags = [tag.strip() for tag in tags_str.replace("|", ",").split(",") if tag.strip()]
                        
                        # Save tags
                        tagger.save_tags(file_path, sidecar_files, tags)
                        stats["tagged"] += 1
                        print(f"Tagged: {fname} ({len(tags)} tags)")
                    else:
                        stats["skipped"] += 1
                        print(f"No tags: {fname}")
                        
                except Exception as e:
                    stats["errors"] += 1
                    print(f"Error tagging {fname}: {e}")
        
        return {
            "status": "success",
            "message": f"Batch tagging complete",
            "directory": directory,
            "stats": stats,
            "device": str(tagger.device)
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Batch tagging failed: {str(e)}")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="STAG API - REST Server for image tagging"
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to bind to (default: 127.0.0.1)"
    )
    parser.add_argument(
        "-p", "--port",
        type=int,
        default=8765,
        help="Port to bind to (default: 8765)"
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload on file changes (development mode)"
    )
    
    args = parser.parse_args()
    
    print(f"Starting STAG API on {args.host}:{args.port}")
    print(f"API documentation available at http://{args.host}:{args.port}/docs")
    
    uvicorn.run(
        "server:app",
        host=args.host,
        port=args.port,
        reload=args.reload
    )

if __name__ == "__main__":
    main()
