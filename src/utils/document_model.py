from pydantic import BaseModel
from typing import Optional

class RAG_Document(BaseModel):
    system: str
    course_id: int
    activity_type: Optional[str] = None
    activity_name: Optional[str] = None
    file: Optional[str] = None
    page: Optional[int] = None
    
    activity_longpage: Optional[int] = None
    activity_pdf: Optional[str] = None
    activity_assign: Optional[int] = None
    activity_wiki: Optional[int] = None
    activity_quiz: Optional[int] = None
    activity_forum: Optional[int] = None
    