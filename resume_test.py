from resume import Resume

r = Resume('resumes/Alvis, Mark.docx')

def test_word_count():
    assert r.getWordCount() > 0

def text_func():
    text = r.getText()
    if text:
        return True

def test_text():
    assert text_func() == True

def test_get_name():
    assert len(r.getName()) > 0

def test_count_jobs():
    assert str(r.countJobs()).isdigit() == True

def test_avg_job_len():
    assert r.avgJobTime() > 0