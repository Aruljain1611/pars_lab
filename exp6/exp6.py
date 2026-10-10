from flask import Flask, request, render_template_string

app = Flask(__name__)

# --- Bayesian Configuration ---

classes = ['Normal', 'Warning', 'Danger']

# Prior Probabilities P(C)
priors = {
    'Normal': 0.60,
    'Warning': 0.30,
    'Danger': 0.10
}

# Likelihoods P(X | C)
likelihoods = {
    'Cool':     {'Normal': 0.60, 'Warning': 0.10, 'Danger': 0.00},
    'Moderate': {'Normal': 0.30, 'Warning': 0.40, 'Danger': 0.10},
    'High':     {'Normal': 0.09, 'Warning': 0.40, 'Danger': 0.30},
    'Critical': {'Normal': 0.01, 'Warning': 0.10, 'Danger': 0.60}
}

# Loss Matrix: loss_matrix[action][true_class]
loss_matrix = {
    'Assume Normal':  {'Normal': 0,   'Warning': 50, 'Danger': 1000},
    'Assume Warning': {'Normal': 20,  'Warning': 0,  'Danger': 200},
    'Assume Danger':  {'Normal': 100, 'Warning': 50, 'Danger': 0}
}

# Action Triggers based on Selected Minimum Risk Class
action_triggers = {
    'Assume Normal':  "🟢 Continue Normal Operation.",
    'Assume Warning': "🟡 Trigger: Issue Maintenance Alert - Schedule check within 24h.",
    'Assume Danger':  "🔴 Trigger: EMERGENCY SHUTDOWN - Halt production immediately!"
}

# --- HTML Template ---
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Bayesian Risk Minimization</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 800px; margin: 40px auto; padding: 20px; background: #f9f9f9; }
        .card { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin-bottom: 20px; }
        h1, h2 { color: #333; }
        table { width: 100%; border-collapse: collapse; margin-top: 10px; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: center; }
        th { background-color: #f2f2f2; }
        .action-box { font-size: 1.2em; font-weight: bold; padding: 15px; border-radius: 5px; color: white; text-align: center;}
        .Normal { background: #2ecc71; }
        .Warning { background: #f1c40f; color: black; }
        .Danger { background: #e74c3c; }
        .note { font-size: 0.9em; color: #555; font-style: italic; }
    </style>
</head>
<body>
    <h1>Industrial Sensor Risk Minimization</h1>

    <div class="card">
        <h2>0. Problem Setup</h2>
        <p><b>Classes:</b> {% for c in classes %}{{ c }}{% if not loop.last %}, {% endif %}{% endfor %}</p>
        <p><b>Actions:</b> {% for a in actions %}{{ a }}{% if not loop.last %}, {% endif %}{% endfor %}</p>
        <h2 style="font-size:1.1em;">Prior Probabilities P(C)</h2>
        <table>
            <tr><th>Class</th><th>Prior</th></tr>
            {% for c in classes %}
            <tr><td>{{ c }}</td><td>{{ priors[c] }}</td></tr>
            {% endfor %}
        </table>
        <h2 style="font-size:1.1em;">Likelihoods P(X | C)</h2>
        <table>
            <tr><th>Sensor Reading (X)</th>{% for c in classes %}<th>{{ c }}</th>{% endfor %}</tr>
            {% for x in readings %}
            <tr><td>{{ x }}</td>{% for c in classes %}<td>{{ likelihoods[x][c] }}</td>{% endfor %}</tr>
            {% endfor %}
        </table>
        <h2 style="font-size:1.1em;">Loss Matrix &lambda;(Action | True Class)</h2>
        <table>
            <tr><th>Action / True Class</th>{% for c in classes %}<th>{{ c }}</th>{% endfor %}</tr>
            {% for a in actions %}
            <tr><td><b>{{ a }}</b></td>{% for c in classes %}<td>{{ loss_matrix[a][c] }}</td>{% endfor %}</tr>
            {% endfor %}
        </table>
    </div>
    
    <div class="card">
        <form method="POST">
            <label for="sensor_reading"><b>Select Temperature Sensor Reading:</b></label>
            <select name="sensor_reading" id="sensor_reading" style="padding:5px; margin-left:10px;">
                <option value="Cool">Cool</option>
                <option value="Moderate">Moderate (Demonstrates Risk vs Probability)</option>
                <option value="High">High</option>
                <option value="Critical">Critical</option>
            </select>
            <button type="submit" style="padding:6px 15px; margin-left:10px; cursor:pointer;">Analyze</button>
        </form>
    </div>

    {% if results %}
    <div class="card">
        <h2>1. Input</h2>
        <p>Observed Sensor Reading: <b>{{ results.input }}</b></p>
    </div>

    <div class="card">
        <h2>2. Posterior Probabilities $P(Class | Input)$</h2>
        <table>
            <tr><th>Class</th><th>Calculated Probability</th></tr>
            {% for cls, prob in results.posteriors.items() %}
            <tr><td>{{ cls }}</td><td>{{ "%.1f"|format(prob * 100) }}%</td></tr>
            {% endfor %}
        </table>
        <p class="note">Notice: Highest probability does not always mean it's the safest choice.</p>
    </div>

    <div class="card">
        <h2>3. Expected Risk (Loss) for Each Action</h2>
        <table>
            <tr><th>Action Considered</th><th>Expected Risk Score</th></tr>
            {% for action, risk in results.risks.items() %}
            <tr {% if action == results.selected_action %}style="background-color:#e8f4f8; font-weight:bold;"{% endif %}>
                <td>{{ action }}</td>
                <td>{{ "%.2f"|format(risk) }}</td>
            </tr>
            {% endfor %}
        </table>
    </div>

    <div class="card">
        <h2>4. Selected Decision & Triggered Action</h2>
        <p>Selected Class Action (Minimum Risk): <b>{{ results.selected_action }}</b></p>
        
        <div class="action-box {{ results.style }}">
            {{ results.triggered_action }}
        </div>
    </div>
    {% endif %}
    
</body>
</html>
"""

@app.route('/', methods=['GET', 'POST'])
def index():
    results = None
    if request.method == 'POST':
        x = request.form['sensor_reading']
        
        # 1. Calculate Evidence P(X) to normalize probabilities
        px = sum([likelihoods[x][c] * priors[c] for c in classes])
        
        # 2. Calculate Posterior Probabilities P(C|X)
        posteriors = {}
        for c in classes:
            posteriors[c] = (likelihoods[x][c] * priors[c]) / px
            
        # 3. Calculate Expected Risk for each possible action
        # Risk = Sum( Loss(Action | TrueClass) * P(TrueClass | X) )
        risks = {}
        for action in loss_matrix.keys():
            expected_risk = 0
            for c in classes:
                expected_risk += loss_matrix[action][c] * posteriors[c]
            risks[action] = expected_risk
            
        # 4. Select Action with MINIMUM Risk
        selected_action = min(risks, key=risks.get)
        
        # Map styling
        style_map = {'Assume Normal': 'Normal', 'Assume Warning': 'Warning', 'Assume Danger': 'Danger'}
        
        results = {
            'input': x,
            'posteriors': posteriors,
            'risks': risks,
            'selected_action': selected_action,
            'triggered_action': action_triggers[selected_action],
            'style': style_map[selected_action]
        }
        
    return render_template_string(HTML_TEMPLATE, results=results,
                                  classes=classes, actions=list(loss_matrix.keys()),
                                  priors=priors, likelihoods=likelihoods,
                                  loss_matrix=loss_matrix,
                                  readings=list(likelihoods.keys()))

if __name__ == '__main__':
    print("Server running at http://127.0.0.1:5000/")
    app.run(debug=True)