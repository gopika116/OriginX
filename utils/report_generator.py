import os
from datetime import datetime

class ReportGenerator:

    @staticmethod
    def generate_pdf_report(
        filename,
        media_type,
        prediction,
        confidence,
        risk,
        influential_region,
        output_path
    ):
        """
        Generate forensic report as a clean PDF or text audit log file.
        """
        try:
            from reportlab.lib.pagesizes import letter
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib import colors

            doc = SimpleDocTemplate(output_path, pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
            story = []
            styles = getSampleStyleSheet()

            title_style = ParagraphStyle(
                'DocTitle',
                parent=styles['Heading1'],
                fontSize=24,
                leading=28,
                textColor=colors.HexColor('#00E5FF'),
                spaceAfter=10
            )

            heading_style = ParagraphStyle(
                'DocHeading',
                parent=styles['Heading2'],
                fontSize=14,
                leading=18,
                textColor=colors.HexColor('#FFFFFF'),
                spaceAfter=6
            )

            normal_style = ParagraphStyle(
                'DocNormal',
                parent=styles['Normal'],
                fontSize=10,
                leading=14,
                textColor=colors.HexColor('#CCCCCC')
            )

            # Title & Header
            story.append(Paragraph("ORIGINX FORENSIC ANALYSIS REPORT", title_style))
            story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')} | System Version: OriginX v2.0", normal_style))
            story.append(Spacer(1, 15))

            # Verdict Box Data
            v_color = colors.HexColor('#FF4A4A') if prediction == 'FAKE' else (colors.HexColor('#4AFF8F') if prediction == 'REAL' else colors.HexColor('#FFAA00'))
            
            data = [
                [Paragraph("<b>Target File:</b>", normal_style), Paragraph(str(filename), normal_style)],
                [Paragraph("<b>Media Type:</b>", normal_style), Paragraph(str(media_type).upper(), normal_style)],
                [Paragraph("<b>Verdict:</b>", normal_style), Paragraph(f"<font color='{v_color.hexval()}'><b>{prediction}</b></font>", normal_style)],
                [Paragraph("<b>Confidence Score:</b>", normal_style), Paragraph(f"{confidence}%", normal_style)],
                [Paragraph("<b>Risk Classification:</b>", normal_style), Paragraph(str(risk), normal_style)],
                [Paragraph("<b>Influential Feature Region:</b>", normal_style), Paragraph(str(influential_region), normal_style)],
            ]

            t = Table(data, colWidths=[150, 380])
            t.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#121620')),
                ('TEXTCOLOR', (0,0), (-1,-1), colors.whitesmoke),
                ('ALIGN', (0,0), (-1,-1), 'LEFT'),
                ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
                ('BOTTOMPADDING', (0,0), (-1,-1), 8),
                ('TOPPADDING', (0,0), (-1,-1), 8),
                ('LEFTPADDING', (0,0), (-1,-1), 12),
                ('RIGHTPADDING', (0,0), (-1,-1), 12),
                ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#2A3245'))
            ]))
            story.append(t)
            story.append(Spacer(1, 20))

            # Audit Summary
            story.append(Paragraph("<b>Forensic Analysis Summary</b>", heading_style))
            summary_text = (
                f"The target digital artifact ({filename}) underwent automated multi-modal analysis using OriginX's "
                f"Vision Transformer and explainable AI architecture. The global probability distributions and local feature "
                f"attributions yielded a verdict of <b>{prediction}</b> with a calculated confidence index of <b>{confidence}%</b>. "
                f"Region attribution indicates the primary model decision evidence was localized around <b>{influential_region}</b>."
            )
            story.append(Paragraph(summary_text, normal_style))
            story.append(Spacer(1, 15))

            # Footer Disclaimer
            story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#2A3245'), spaceAfter=10))
            disclaimer = "CONFIDENTIAL FORENSIC REPORT - Generated automatically by OriginX Deepfake Detection Engine."
            story.append(Paragraph(disclaimer, ParagraphStyle('Disc', parent=normal_style, fontSize=8, textColor=colors.HexColor('#666666'))))

            doc.build(story)
            return output_path

        except ImportError:
            # Fallback text audit report if reportlab is not installed
            txt_path = output_path.replace(".pdf", ".txt")
            with open(txt_path, "w") as f:
                f.write("="*60 + "\n")
                f.write("ORIGINX FORENSIC ANALYSIS AUDIT REPORT\n")
                f.write("="*60 + "\n\n")
                f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}\n")
                f.write(f"Target File: {filename}\n")
                f.write(f"Media Type: {media_type}\n")
                f.write(f"Prediction: {prediction}\n")
                f.write(f"Confidence: {confidence}%\n")
                f.write(f"Risk Level: {risk}\n")
                f.write(f"Influential Region: {influential_region}\n\n")
                f.write("Report generated by OriginX Deepfake Detection System.\n")
                f.write("="*60 + "\n")
            return txt_path
