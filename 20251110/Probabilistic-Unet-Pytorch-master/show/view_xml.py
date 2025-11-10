import xml.etree.ElementTree as ET

# 修改为你的 XML 文件路径
xml_path = r"E:\workspace\LIDC-IDRI\processingCT_manifest_1760698817279\LIDC-IDRI\LIDC-IDRI-0005\01-01-2000-NA-NA-42125\3000548.000000-NA-86225\076.xml"

# 解析 XML
tree = ET.parse(xml_path)
root = tree.getroot()
ns = {'lidc': root.tag.split('}')[0].strip('{')}  # 命名空间自动提取

# 遍历医生的 readingSession
for i, rs in enumerate(root.findall('.//lidc:readingSession', ns), start=1):
    print(f"\n=== 🩺 医生 {i} 的标注信息 ===")
    # 每位医生的结节标注
    for nodule in rs.findall('lidc:unblindedReadNodule', ns):
        nodule_id = nodule.findtext('lidc:noduleID', default='', namespaces=ns)
        print(f"  🔹 结节 ID: {nodule_id}")

        # 结节特征（恶性度等）
        ch = nodule.find('lidc:characteristics', ns)
        if ch is not None:
            malignancy = ch.findtext('lidc:malignancy', default='', namespaces=ns)
            print(f"    └─ 恶性程度 (malignancy): {malignancy or '缺失'}")

        # 每个 ROI 对应一张切片
        for j, roi in enumerate(nodule.findall('lidc:roi', ns), start=1):
            sop = roi.findtext('lidc:imageSOP_UID', default='', namespaces=ns)
            zpos = roi.findtext('lidc:imageZposition', default='', namespaces=ns)
            edge_maps = roi.findall('lidc:edgeMap', ns)
            print(f"    ROI {j}: 切片Z={zpos}, 点数={len(edge_maps)}, SOP_UID={sop[:40]}...")

            # 若只想查看前几个点坐标，可取消下一行注释：
            # for em in edge_maps[:3]:
            #     x = em.findtext('lidc:xCoord', default='', namespaces=ns)
            #     y = em.findtext('lidc:yCoord', default='', namespaces=ns)
            #     print(f"       → ({x}, {y})")

    # 若有非结节区域：
    non_nodules = rs.findall('lidc:nonNodule', ns)
    if non_nodules:
        print(f"  ⚪ 非结节标注数量: {len(non_nodules)}")
