"""Product-name ordering: growth leads the name with the fragment."""
from asatro.engine.thompson_sampling import ThompsonSampler


class _R:
    def __init__(self, name):
        self.reagent_name = name


def _reagents():
    # flat route order for amide -> suzuki: acid (comp0), fragment (comp1), boronic
    return [_R("1910687"), _R("FRAG"), _R("7375753")]


def test_default_order_is_route_order():
    ts = ThompsonSampler()
    assert ts._product_name(_reagents()) == "1910687_FRAG_7375753"


def test_fragment_leads_when_lead_index_set():
    ts = ThompsonSampler()
    ts.name_lead_index = 1  # the fragment's flat slot
    # fragment first, the rest keep route order (step-1 acid, then step-2 boronic)
    assert ts._product_name(_reagents()) == "FRAG_1910687_7375753"


def test_lead_index_zero_is_a_noop():
    ts = ThompsonSampler()
    ts.name_lead_index = 0
    assert ts._product_name(_reagents()) == "1910687_FRAG_7375753"


def test_out_of_range_lead_index_ignored():
    ts = ThompsonSampler()
    ts.name_lead_index = 9
    assert ts._product_name(_reagents()) == "1910687_FRAG_7375753"
