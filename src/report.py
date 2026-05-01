from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet

def generate_pdf(data_summary, filename="report.pdf"):
    doc = SimpleDocTemplate(filename)
    styles = getSampleStyleSheet()

    content = []

    content.append(Paragraph("AutoInsights Report", styles["Title"]))
    content.append(Paragraph("<br/>", styles["Normal"]))

    for line in data_summary:
        content.append(Paragraph(line, styles["Normal"]))

    doc.build(content)

    return filename