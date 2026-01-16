# STAG Bug Fix Suggestions

## Problem: Tagging process hangs/freezes without error output

### Root Causes Identified:

1. **No timeout for individual image processing**
   - GPU/CUDA inference can hang indefinitely
   - No way to recover from stuck operations

2. **Insufficient error handling**
   - Exceptions are caught but not logged properly
   - No stack traces for debugging

3. **Potential memory leaks**
   - Images not explicitly closed after processing
   - Can cause OOM with large directories

4. **GUI freezing**
   - No periodic UI updates during processing
   - User gets no feedback that processing is ongoing

### Recommended Fixes:

#### Fix 1: Add timeout to image processing

```python
import signal
from contextlib import contextmanager

@contextmanager
def timeout(seconds):
    def timeout_handler(signum, frame):
        raise TimeoutError(f"Operation timed out after {seconds} seconds")
    
    # Set alarm
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)

# In get_tags_for_image():
def get_tags_for_image(self, pil_image: Image.Image) -> str:
    try:
        with timeout(30):  # 30 second timeout per image
            torch_image = self.transform(pil_image).unsqueeze(0).to(self.device)
            res = inference(torch_image, self.model)
            return res[0]
    except TimeoutError as e:
        print(f"Tagging timeout: {e}")
        return ""
    except Exception as e:
        import traceback
        print(f"Tagging failed: {e}")
        print(traceback.format_exc())
        return ""
```

#### Fix 2: Explicitly close images

```python
def enter_dir(self, img_dir: str, stop_event: threading.Event) -> None:
    for current_dir, _, file_list in os.walk(img_dir):
        for fname in sorted(file_list):
            # ... existing code ...
            
            image, loader = self.load_image(image_file)
            
            if image is not None:
                try:
                    print(f'Looking at {image_file} loaded with {loader}:')
                    tag_string = self.get_tags_for_image(image)
                    tags = [item.strip() for item in tag_string.split("|")]
                    print(f"Tags found: {tags}")
                    self.save_tags(image_file, sidecar_files, tags)
                finally:
                    # IMPORTANT: Close image to free memory
                    image.close()
```

#### Fix 3: Add progress callback for GUI

```python
# In SKTagger class:
def __init__(self, ..., progress_callback=None):
    # ... existing code ...
    self.progress_callback = progress_callback

def enter_dir(self, img_dir: str, stop_event: threading.Event) -> None:
    total_files = sum(len(files) for _, _, files in os.walk(img_dir))
    processed = 0
    
    for current_dir, _, file_list in os.walk(img_dir):
        for fname in sorted(file_list):
            # ... existing processing ...
            
            processed += 1
            if self.progress_callback:
                self.progress_callback(processed, total_files, fname)
```

#### Fix 4: Add periodic GUI updates

```python
# In stag_gui.py run_tagger_thread():
def update_progress(current, total, filename):
    progress_pct = (current / total) * 100 if total > 0 else 0
    self.root.after(0, lambda: self.progress_label.config(
        text=f"Processing: {current}/{total} ({progress_pct:.1f}%) - {filename}"
    ))

tagger = SKTagger(..., progress_callback=update_progress)
```

#### Fix 5: Add watchdog for complete hang detection

```python
import time
from threading import Thread

class WatchdogTimer:
    def __init__(self, timeout, callback):
        self.timeout = timeout
        self.callback = callback
        self.last_activity = time.time()
        self.running = True
        self.thread = Thread(target=self._watch, daemon=True)
        self.thread.start()
    
    def reset(self):
        self.last_activity = time.time()
    
    def stop(self):
        self.running = False
    
    def _watch(self):
        while self.running:
            time.sleep(1)
            if time.time() - self.last_activity > self.timeout:
                self.callback()
                break

# Usage in enter_dir():
def watchdog_triggered():
    print("ERROR: Processing appears to be stuck!")
    print("Last file may have caused a hang. Consider skipping it.")
    self.stop_event.set()

watchdog = WatchdogTimer(timeout=60, callback=watchdog_triggered)

for fname in sorted(file_list):
    watchdog.reset()  # Reset on each file
    # ... process file ...
    
watchdog.stop()
```

### Testing Recommendations:

1. Test with a known "problematic" image that causes hangs
2. Test with very large directories (1000+ images)
3. Monitor memory usage during processing
4. Test cancellation at various stages
5. Test with corrupted/invalid image files

### Quick Fix (Minimal Change):

Add this to the beginning of `get_tags_for_image()`:

```python
def get_tags_for_image(self, pil_image: Image.Image) -> str:
    import signal
    
    def alarm_handler(signum, frame):
        raise TimeoutError("Image processing timeout")
    
    signal.signal(signal.SIGALRM, alarm_handler)
    signal.alarm(30)  # 30 second timeout
    
    try:
        torch_image = self.transform(pil_image).unsqueeze(0).to(self.device)
        res = inference(torch_image, self.model)
        signal.alarm(0)  # Cancel alarm
        return res[0]
    except TimeoutError:
        signal.alarm(0)
        print(f"Image processing timed out")
        return ""
    except Exception as e:
        signal.alarm(0)
        print(f"Tagging failed: {e}")
        import traceback
        traceback.print_exc()
        return ""
```

Note: `signal.alarm()` only works on Unix/Linux. For Windows, use `threading.Timer` instead.
