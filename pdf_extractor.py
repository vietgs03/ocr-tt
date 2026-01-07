import fitz  # PyMuPDF
import os
from pathlib import Path

def pdf_to_images(pdf_path, output_folder="pdf_images", dpi=150):
    """
    Extract all pages from PDF as images
    
    Args:
        pdf_path: Path to PDF file
        output_folder: Folder to save images
        dpi: Resolution (150 is good balance between quality and size)
    
    Returns:
        List of image paths
    """
    # Create output folder
    os.makedirs(output_folder, exist_ok=True)
    
    # Get PDF filename without extension
    pdf_name = Path(pdf_path).stem
    
    # Open PDF
    doc = fitz.open(pdf_path)
    image_paths = []
    
    print(f"📄 Processing: {pdf_path}")
    print(f"📖 Total pages: {len(doc)}")
    
    # Extract each page
    for page_num in range(len(doc)):
        page = doc[page_num]
        
        # Convert to image
        # zoom factor = dpi / 72 (72 is default PDF DPI)
        zoom = dpi / 72
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)
        
        # Save as PNG
        output_path = os.path.join(
            output_folder, 
            f"{pdf_name}_page_{page_num + 1:03d}.png"
        )
        pix.save(output_path)
        image_paths.append(output_path)
        
        print(f"  ✅ Page {page_num + 1}/{len(doc)} → {output_path}")
    
    doc.close()
    print(f"🎉 Extracted {len(image_paths)} pages to {output_folder}/\n")
    
    return image_paths

if __name__ == "__main__":
    # Process both PDFs
    pdf_files = [
        "dongdau6016_06-07-2023-14-56-28_bc-hoi-dong_0001.pdf",
        "bao-cao-ket-qua-thuc-hien-chuong-trinh-giam-sat-nam-2022-cua-tt-hdnd_trongtb-11-07-2023_08h54p4411.07.2023_14h31p10_signed.pdf"
    ]
    
    all_images = []
    
    for pdf_file in pdf_files:
        if os.path.exists(pdf_file):
            images = pdf_to_images(pdf_file, output_folder="test_images", dpi=150)
            all_images.extend(images)
        else:
            print(f"⚠️ File not found: {pdf_file}")
    
    print("="*60)
    print(f"✅ Total images extracted: {len(all_images)}")
    print(f"📁 Output folder: test_images/")
    print("="*60)
    print("\nReady for batch OCR processing!")
