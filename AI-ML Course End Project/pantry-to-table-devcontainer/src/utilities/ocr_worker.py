"""
Standalone PaddleOCR worker.

This absolutely not mine.
This was taken directly from Claude/Gemini conversations when trying
to fix library collisions between YoloV8 and PaddleOCR libraries.
Fix is to isolate Paddle into it's own processs.

Runs in its own Python process to avoid MKL conflicts with PyTorch/ultralytics.
Invoked as a subprocess from OCRIngredientLabelUtilities.

Usage:
    python ocr_worker.py <image_path>

Output (stdout):
    JSON array: [{"text": str, "confidence": float, "box": [[x,y], ...]}, ...]

Errors go to stderr, exit code non-zero on failure.

        THIS DOES NOT WORK!! ISSUES WITH LIBRARY COLLISION AND DEPENDENCIES 
        - Paddle and PyTorch's bundled MKL runtimes conflict, crashing paddle at inference.
        - NGC's global LD_LIBRARY_PATH leaks into subprocesses, so isolation didn't help.

        Long term fix is to create a dedicated Paddle devcontainer and host it in a webservice
"""

import sys
import json

from paddleocr import PaddleOCR


def run_ocr(image_path: str) -> list:
    """Run PaddleOCR on the given image and return a flat list of results."""
    ocr = PaddleOCR(
        use_textline_orientation=True,
        lang='en',
        device='cpu',
    )

    raw_results = ocr.ocr(image_path)

    # PaddleOCR 3.x returns list of OCRResult objects
    output = []
    for result in raw_results:
        # 3.x dict-style access
        if hasattr(result, 'get') or isinstance(result, dict):
            texts = result.get('rec_texts', [])
            scores = result.get('rec_scores', [])
            polys = result.get('rec_polys', [])
            for text, score, poly in zip(texts, scores, polys):
                output.append({
                    'text': text,
                    'confidence': float(score),
                    'box': [[float(p[0]), float(p[1])] for p in poly],
                })
        else:
            # Fallback for 2.x-style nested list [[box, (text, score)], ...]
            if result is None:
                continue
            for line in result:
                box = line[0]
                text = line[1][0]
                score = line[1][1]
                output.append({
                    'text': text,
                    'confidence': float(score),
                    'box': [[float(p[0]), float(p[1])] for p in box],
                })

    return output


def main():
    if len(sys.argv) != 2:
        print('Usage: ocr_worker.py <image_path>', file=sys.stderr)
        sys.exit(2)

    image_path = sys.argv[1]
    try:
        results = run_ocr(image_path)
        # Write JSON to stdout — must be the only thing printed to stdout
        sys.stdout.write(json.dumps(results))
        sys.stdout.flush()
    except Exception as e:
        print(f'OCR worker error: {type(e).__name__}: {e}', file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
