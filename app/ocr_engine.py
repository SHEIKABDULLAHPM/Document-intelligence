import pytesseract
from PIL import Image
import cv2
import numpy as np
from pdf2image import convert_from_path
import pypdfium2 as pdfium
import io

class OCREngine:
    """
    OCR Engine using Tesseract with image preprocessing
    """
    def __init__(self, config=None):
        self.config = config or '--oem 3 --psm 6'
        # OEM 3: Default, based on what is available
        # PSM 6: Assume a single uniform block of text
        
    def preprocess_image(self, image):
        """
        Preprocess image for better OCR accuracy
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        # Apply threshold to get binary image
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Denoise
        denoised = cv2.fastNlMeansDenoising(thresh)
        
        # Deskew (optional - for skewed documents)
        # ...
        
        return denoised
    
    def extract_text_from_image(self, image_path):
        """
        Extract text from image file
        """
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
            
        # Preprocess
        processed = self.preprocess_image(image)
        
        # Perform OCR
        text = pytesseract.image_to_string(processed, config=self.config)
        
        return text
    
    def extract_text_from_pdf(self, pdf_path):
        """
        Extract text from PDF (handles both text-based and scanned PDFs)
        """
        text = ""
        
        try:
            # Try to extract text directly (for text-based PDFs)
            import PyPDF2
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                for page in reader.pages:
                    text += page.extract_text() + "\n"
                    
            # If text is minimal, it might be scanned - use OCR
            if len(text.strip()) < 100:  # Threshold for text-based vs scanned
                text = self._ocr_pdf_scanned(pdf_path)
                
        except Exception as e:
            print(f"PDF text extraction failed: {e}")
            text = self._ocr_pdf_scanned(pdf_path)
            
        return text
    
    def _ocr_pdf_scanned(self, pdf_path):
        """
        OCR for scanned PDFs
        """
        # Convert PDF to images
        images = convert_from_path(pdf_path, dpi=300)
        
        text = ""
        for i, image in enumerate(images):
            # Convert PIL to numpy array
            img_array = np.array(image)
            processed = self.preprocess_image(img_array)
            page_text = pytesseract.image_to_string(processed, config=self.config)
            text += f"Page {i+1}:\n{page_text}\n"
            
        return text