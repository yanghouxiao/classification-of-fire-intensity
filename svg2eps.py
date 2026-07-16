# # svg2eps.py
#
# import os
# import tempfile
# import xml.etree.ElementTree as ET
#
# from svglib.svglib import svg2rlg
# from reportlab.graphics import renderPS
#
# # 输入输出路径
# svg_file = r"C:\Users\yanghouxiao\Desktop\F6.svg"
# eps_file = r"C:\Users\yanghouxiao\Desktop\F6.eps"
#
# # 读取 SVG
# tree = ET.parse(svg_file)
# root = tree.getroot()
#
# # 替换字体
# for elem in root.iter():
#
#     if 'font-family' in elem.attrib:
#         elem.attrib['font-family'] = 'Helvetica'
#
#     if 'style' in elem.attrib:
#         style = elem.attrib['style']
#
#         style = style.replace('TimesNewRomanPSMT', 'Helvetica')
#         style = style.replace('Times New Roman', 'Helvetica')
#         style = style.replace('ArialMT', 'Helvetica')
#         style = style.replace('Arial', 'Helvetica')
#         style = style.replace('Calibri', 'Helvetica')
#
#         elem.attrib['style'] = style
#
# # 创建临时 svg
# fd, temp_svg_path = tempfile.mkstemp(suffix=".svg")
# os.close(fd)
#
# # 保存
# tree.write(temp_svg_path)
#
# # 转换
# drawing = svg2rlg(temp_svg_path)
# renderPS.drawToFile(drawing, eps_file)
#
# print("转换成功！")
# print("EPS 文件位置：")
# print(eps_file)

from PIL import Image, ImageDraw, ImageFont

img = Image.open(r"C:\Users\yanghouxiao\Desktop\F11.png")
draw = ImageDraw.Draw(img)

# 位置自己微调
draw.text((50, 50), "(a)", fill="white")
draw.text((400, 50), "(b)", fill="white")
draw.text((750, 50), "(c)", fill="white")

img.save("F11_labeled.png")





