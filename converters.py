
import json
import jsonlines

from langchain_core.documents.base import Document
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions
)


def metadata(document, metadata):
    metadata["source"] = document.get("source")
    metadata["page_no"] = document.get("page_no")
    return metadata


def extract_metadata(documents):
    """
    Extract metadata from a list of documents.

    Args:
        documents (list[Document]): A list of Document objects.

    Returns:
        list[Document]: A list of Document objects with the extracted metadata.
    """
    return [
        Document(
            page_content=doc.page_content.encode('utf8'),
            metadata={
                "source": doc.metadata['source'],
                "page_no": doc.metadata['dl_meta']['doc_items'][0]['prov'][0]['page_no'],
            }
        )
        for doc in documents
    ]


def pdf_converter(OCR=False):
    """
    Configures and returns a DocumentConverter instance for processing PDF documents.

    Args:
        OCR (bool, optional): Determines whether Optical Character Recognition (OCR) should be performed on the PDF documents.

    Returns:
        DocumentConverter: A configured DocumentConverter instance for processing PDF documents.
    """
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_table_structure = True
    pipeline_options.table_structure_options.do_cell_matching = True

    if OCR:
        pipeline_options.do_ocr = True
        pipeline_options.ocr_options.lang = ["es"]
        pipeline_options.ocr_options.force_full_page_ocr = True

    pipeline_options.accelerator_options = AcceleratorOptions(
            num_threads=4, device=AcceleratorDevice.AUTO
        )

    doc_converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )

    return doc_converter


def save_docs_to_jsonl(documents, jsonl_file_path):
    """
    Saves a list of documents to a JSONL (JSON Lines) file.

    Args:
        documents (list): A list of document objects to be saved.
        jsonl_file_path (str): The file path where the JSONL file will be saved.

    Returns:
        None
    """
    with jsonlines.open(jsonl_file_path, mode="w") as writer:
        for doc in documents:
            doc_dump = doc.model_dump()
            page_content = doc_dump["page_content"]
            metadata = doc_dump["metadata"]
            to_jsonl = {
                "page_content": page_content,
                "page_no": metadata["page_no"],
                "source": metadata["source"]
            }
            writer.write(to_jsonl)


