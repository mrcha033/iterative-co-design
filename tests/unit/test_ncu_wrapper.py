from icd.measure.ncu_wrapper import parse_l2_hit_from_section_json


def test_parse_l2_hit_from_section_json(tmp_path):
    p = tmp_path / "ncu.json"
    p.write_text('{"l2_tex__t_sector_hit_rate.pct": 87.5}')
    res = parse_l2_hit_from_section_json(str(p))
    assert "l2_hit_pct" in res and abs(res["l2_hit_pct"] - 87.5) < 1e-6

