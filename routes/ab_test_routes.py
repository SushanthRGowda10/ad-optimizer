from flask import Blueprint, render_template, request

ab_test_bp = Blueprint("ab_test", __name__)

#@ab_test_bp.route("/ab-test", methods=["GET", "POST"])
def ab_test():
    result = None
    winner = None
    ctr_a = ""
    ctr_b = ""

    if request.method == "POST":
        ctr_a = request.form.get("ctr_a", "").strip()
        ctr_b = request.form.get("ctr_b", "").strip()

        try:
            a = float(ctr_a)
            b = float(ctr_b)

            if a > b:
                result = f"Ad A performs better ({a}% vs {b}%)."
                winner = "A"
            elif b > a:
                result = f"Ad B performs better ({b}% vs {a}%)."
                winner = "B"
            else:
                result = f"Both ads perform equally ({a}%)."
                winner = "Tie"

        except ValueError:
            result = "Invalid input. Enter numbers only."

    return render_template(
        "ab_test.html",
        result=result,
        winner=winner,
        ctr_a=ctr_a,
        ctr_b=ctr_b,
    )