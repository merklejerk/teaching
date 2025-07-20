<script lang="ts">
	import { onMount } from 'svelte';
	import Chart from 'chart.js/auto';

	// Parameter ranges
	const W_MIN = -2;
	const W_MAX = 2;
	const B_MIN = -2;
	const B_MAX = 2;
	const X_MIN = -10;
	const X_MAX = 10;
	const Y_TRUE_MIN = 0;
	const Y_TRUE_MAX = 1;

	// Gradient descent settings
	let learningRate = 0.1;
	let isAutoStepping = false;
	let autoStepInterval: ReturnType<typeof setInterval>;

	// NN parameters
	let w = 0.5;
	let b = 0.1;

	// Input
	let x = 2.5;

	// True value
	let y_true = 0.9;

	let chartElement: HTMLCanvasElement;
	let lossChartElement: HTMLCanvasElement;
	let nnOutputChartElement: HTMLCanvasElement;
	let chart: Chart;
	let lossChart: Chart;
	let nnOutputChart: Chart;

	// Sigmoid function
	const sigmoid = (val: number) => 1 / (1 + Math.exp(-val));

	// Neural network output
	$: y = sigmoid(w * x + b);

	// Loss function (Mean Squared Error)
	$: loss = Math.pow(y_true - y, 2);

	// Partial derivatives
	$: dL_dw = -2 * (y_true - y) * y * (1 - y) * x;
	$: dL_db = -2 * (y_true - y) * y * (1 - y);

	const randomizeParameters = () => {
		w = Math.random() * (W_MAX - W_MIN) + W_MIN;
		b = Math.random() * (B_MAX - B_MIN) + B_MIN;
		x = Math.random() * (X_MAX - X_MIN) + X_MIN;
		y_true = Math.random() * (Y_TRUE_MAX - Y_TRUE_MIN) + Y_TRUE_MIN;
	};

	const performGradientDescentStep = () => {
		// Update parameters using gradient descent: param = param - learning_rate * gradient
		w = w - learningRate * dL_dw;
		b = b - learningRate * dL_db;
		
		// Clamp values to stay within bounds
		w = Math.max(W_MIN, Math.min(W_MAX, w));
		b = Math.max(B_MIN, Math.min(B_MAX, b));
	};

	const toggleAutoStepping = () => {
		if (isAutoStepping) {
			clearInterval(autoStepInterval);
			isAutoStepping = false;
		} else {
			isAutoStepping = true;
			autoStepInterval = setInterval(performGradientDescentStep, 500); // Step every 500ms
		}
	};

	const resetParameters = () => {
		w = 0.5;
		b = 0.1;
		if (isAutoStepping) {
			toggleAutoStepping();
		}
	};

	const updateCharts = () => {
		if (chart && lossChart && nnOutputChart) {
			// Update derivative and loss charts
			const dL_dw_data = [];
			const dL_db_data = [];
			const loss_data_w = [];
			const loss_data_b = [];

			for (let i = W_MIN; i <= W_MAX; i += 0.1) {
				// For change in w
				const temp_w = w + i;
				const temp_y_w = sigmoid(temp_w * x + b);
				const temp_loss_w = Math.pow(y_true - temp_y_w, 2);
				const temp_dL_dw = -2 * (y_true - temp_y_w) * temp_y_w * (1 - temp_y_w) * x;
				loss_data_w.push({ x: i, y: temp_loss_w });
				dL_dw_data.push({ x: i, y: temp_dL_dw });

				// For change in b
				const temp_b = b + i;
				const temp_y_b = sigmoid(w * x + temp_b);
				const temp_loss_b = Math.pow(y_true - temp_y_b, 2);
				const temp_dL_db = -2 * (y_true - temp_y_b) * temp_y_b * (1 - temp_y_b);
				loss_data_b.push({ x: i, y: temp_loss_b });
				dL_db_data.push({ x: i, y: temp_dL_db });
			}
			chart.data.datasets[0].data = dL_dw_data;
			chart.data.datasets[1].data = dL_db_data;
			chart.update();

			lossChart.data.datasets[0].data = loss_data_w;
			lossChart.data.datasets[1].data = loss_data_b;
			lossChart.update();

			// Update NN Output chart (vs. input x)
			const nn_output_data = [];
			// Use a fixed range for x to show the effect of w and b on the sigmoid shape
			for (let i = X_MIN; i <= X_MAX; i += 0.2) {
				const current_x = i;
				const temp_y = sigmoid(w * current_x + b);
				nn_output_data.push({ x: current_x, y: temp_y });
			}
			nnOutputChart.data.datasets[0].data = nn_output_data; // The function curve
			nnOutputChart.data.datasets[1].data = [{ x: x, y: y_true }]; // The target point
			nnOutputChart.data.datasets[2].data = [{ x: x, y: y }]; // The prediction point
			nnOutputChart.update();
		}
	};

	$: if (chart && lossChart && nnOutputChart) {
		w, b, x, y_true;
		updateCharts();
	}

	onMount(() => {
		const ctx = chartElement.getContext('2d');
		const lossCtx = lossChartElement.getContext('2d');
		const nnOutputCtx = nnOutputChartElement.getContext('2d');
		if (ctx && lossCtx && nnOutputCtx) {
			chart = new Chart(ctx, {
				type: 'line',
				data: {
					datasets: [
						{
							label: 'dL/dw',
							data: [],
							borderColor: 'red',
							fill: false
						},
						{
							label: 'dL/db',
							data: [],
							borderColor: 'blue',
							fill: false
						}
					]
				},
				options: {
					animation: false,
					responsive: true,
					maintainAspectRatio: false,
					plugins: {
						title: {
							display: true,
							text: 'Partial Derivatives vs. Change in Parameters'
						}
					},
					scales: {
						x: {
							type: 'linear',
							title: {
								display: true,
								text: 'Change in w or b'
							}
						},
						y: {
							title: {
								display: true,
								text: 'Partial Derivative Value'
							}
						}
					}
				}
			});

			lossChart = new Chart(lossCtx, {
				type: 'line',
				data: {
					datasets: [
						{
							label: 'Loss vs Δw',
							data: [],
							borderColor: 'purple',
							fill: false
						},
						{
							label: 'Loss vs Δb',
							data: [],
							borderColor: 'deeppink',
							fill: false
						}
					]
				},
				options: {
					animation: false,
					responsive: true,
					maintainAspectRatio: false,
					plugins: {
						title: {
							display: true,
							text: 'Loss vs. Change in Parameters'
						}
					},
					scales: {
						x: {
							type: 'linear',
							title: {
								display: true,
								text: 'Change in w or b'
							}
						},
						y: {
							title: {
								display: true,
								text: 'Loss Value'
							}
						}
					}
				}
			});

			nnOutputChart = new Chart(nnOutputCtx, {
				type: 'line',
				data: {
					labels: [],
					datasets: [
						{
							label: 'NN Output (y)',
							data: [],
							borderColor: 'orange',
							fill: false,
							pointRadius: 0 // Hide points on the main line
						},
						{
							label: 'Target (x, y_true)',
							data: [],
							backgroundColor: 'green',
							pointRadius: 6,
							type: 'scatter'
						},
						{
							label: 'Prediction (x, y)',
							data: [],
							backgroundColor: 'red',
							pointRadius: 6,
							type: 'scatter'
						}
					]
				},
				options: {
					animation: false,
					responsive: true,
					maintainAspectRatio: false,
					plugins: {
						title: {
							display: true,
							text: 'NN Output vs. Input x'
						}
					},
					scales: {
						x: {
							type: 'linear',
							title: {
								display: true,
								text: 'Input x'
							}
						},
						y: {
							type: 'linear',
							display: true,
							position: 'left',
							title: {
								display: true,
								text: 'Output Value'
							},
							min: 0,
							max: 1
						}
					}
				}
			});
			updateCharts();
		}
	});
</script>

<main>
	<h1>Neural Network Gradient Visualization</h1>

	<div class="network-diagram">
		<div class="layer">
			<div class="node">
				x
				<div class="value">{x.toFixed(2)}</div>
			</div>
		</div>
		<div class="arrows">
			<svg>
				<line x1="0" y1="75" x2="100" y2="75" stroke="black" />
				<text x="50" y="65" text-anchor="middle">w: {w.toFixed(2)}</text>
			</svg>
		</div>
		<div class="layer">
			<div class="node neuron">
				Neuron
				<div class="bias">+ b: {b.toFixed(2)}</div>
			</div>
		</div>
		<div class="arrows">
			<svg>
				<line x1="0" y1="75" x2="100" y2="75" stroke="black" />
			</svg>
		</div>
		<div class="layer">
			<div class="node">
				y
				<div class="value">{y.toFixed(4)}</div>
			</div>
		</div>
	</div>

	<div class="controls">
		<div>
			<label for="w">w: {w.toFixed(2)}</label>
			<input id="w" type="range" bind:value={w} min={W_MIN} max={W_MAX} step="0.01" />
		</div>
		<div>
			<label for="b">b: {b.toFixed(2)}</label>
			<input id="b" type="range" bind:value={b} min={B_MIN} max={B_MAX} step="0.01" />
		</div>
		<div>
			<label for="x">x: {x.toFixed(2)}</label>
			<input id="x" type="range" bind:value={x} min={X_MIN} max={X_MAX} step="0.1" />
		</div>
		<div>
			<label for="y_true">y_true: {y_true.toFixed(2)}</label>
			<input id="y_true" type="range" bind:value={y_true} min={Y_TRUE_MIN} max={Y_TRUE_MAX} step="0.01" />
		</div>
		<div>
			<button on:click={randomizeParameters}>Randomize Parameters</button>
		</div>
	</div>

	<div class="info">
		<p class="formula">
			Cost Function: L = (y<sub>true</sub> - y)<sup>2</sup> = (y<sub>true</sub> - σ(w*x +
			b))<sup>2</sup>
		</p>
		<div class="info-grid">
			<div>
				<p>y = {y.toFixed(4)}</p>
			</div>
			<div>
				<p>Loss = {loss.toFixed(4)}</p>
			</div>
		</div>
		<div class="formula-grid">
			<div>
				<p>∂L/∂w = -2(y<sub>true</sub> - y) * y(1-y) * x</p>
				<p class="value">{dL_dw.toFixed(4)}</p>
			</div>
			<div>
				<p>∂L/∂b = -2(y<sub>true</sub> - y) * y(1-y)</p>
				<p class="value">{dL_db.toFixed(4)}</p>
			</div>
		</div>
	</div>

	<details class="derivations">
		<summary>Show Derivations</summary>
		<div class="derivation-content">
			<h4>Core Components</h4>
			<p>
				<b>Sigmoid Function (σ):</b> The activation function used by the neuron. It squashes any
				input value to a range between 0 and 1.
			</p>
			<p class="formula">σ(z) = 1 / (1 + e<sup>-z</sup>)</p>

			<h4>Chain Rule Application</h4>
			<p>
				We use the chain rule to find the partial derivatives of the loss <code>L</code> with respect to each
				parameter (w, b). The general form is:
			</p>
			<p class="formula">∂L/∂w = (∂L/∂y) * (∂y/∂z) * (∂z/∂w)</p>
			<p>Where:</p>
			<ul>
				<li><code>z = w*x + b</code> (the linear combination)</li>
				<li><code>y = σ(z)</code> (the sigmoid activation)</li>
				<li><code>L = (y_true - y)²</code> (the loss function)</li>
			</ul>

			<h4>Step 1: Derivative of Loss w.r.t. Output (∂L/∂y)</h4>
			<p class="formula">∂L/∂y = 2 * (y<sub>true</sub> - y) * (-1) = -2(y<sub>true</sub> - y)</p>

			<h4>Step 2: Derivative of Activation w.r.t. Linear Input (∂y/∂z)</h4>
			<p>The derivative of the sigmoid function <code>σ(z)</code> is <code>σ(z) * (1 - σ(z))</code>.</p>
			<p class="formula">∂y/∂z = y * (1 - y)</p>

			<h4>Step 3: Derivative of Linear Input w.r.t. Parameters</h4>
			<p class="formula">∂z/∂w = x</p>
			<p class="formula">∂z/∂b = 1</p>

			<h4>Combining the parts:</h4>
			<p><b>For w:</b></p>
			<p class="formula">∂L/∂w = [-2(y<sub>true</sub> - y)] * [y(1-y)] * [x]</p>
			<p><b>For b:</b></p>
			<p class="formula">∂L/∂b = [-2(y<sub>true</sub> - y)] * [y(1-y)] * [1]</p>
		</div>
	</details>

	<details class="gradient-descent">
		<summary>Gradient Descent Controls</summary>
		<div class="gradient-descent-content">
			<div class="gd-controls">
				<div>
					<label for="learningRate">Learning Rate: {learningRate.toFixed(3)}</label>
					<input id="learningRate" type="range" bind:value={learningRate} min="0.001" max="1" step="0.001" />
				</div>
				<div class="gd-buttons">
					<button on:click={performGradientDescentStep}>Single Step</button>
					<button on:click={toggleAutoStepping} class={isAutoStepping ? 'stop' : 'start'}>
						{isAutoStepping ? 'Stop Auto-Step' : 'Start Auto-Step'}
					</button>
					<button on:click={resetParameters}>Reset to Default</button>
				</div>
			</div>
			
			<div class="gradient-step-info">
				<h4>Next Gradient Descent Step (with learning rate {learningRate.toFixed(3)}):</h4>
				<div class="step-grid">
					<div>
						<p>w_new = w - α × ∂L/∂w</p>
						<p class="step-value">
							{w.toFixed(4)} - {learningRate.toFixed(3)} × {dL_dw.toFixed(4)} = 
							<span class="next-value">{(w - learningRate * dL_dw).toFixed(4)}</span>
						</p>
					</div>
					<div>
						<p>b_new = b - α × ∂L/∂b</p>
						<p class="step-value">
							{b.toFixed(4)} - {learningRate.toFixed(3)} × {dL_db.toFixed(4)} = 
							<span class="next-value">{(b - learningRate * dL_db).toFixed(4)}</span>
						</p>
					</div>
				</div>
			</div>
		</div>
	</details>

	<div class="chart-grid">
		<div class="chart-container">
			<canvas bind:this={chartElement}></canvas>
		</div>
		<div class="chart-container">
			<canvas bind:this={lossChartElement}></canvas>
		</div>
		<div class="chart-container">
			<canvas bind:this={nnOutputChartElement}></canvas>
		</div>
	</div>
</main>

<style>
	main {
		padding: 1em;
		max-width: 1400px;
		margin: 0 auto;
		font-family: sans-serif;
	}
	h1 {
		text-align: center;
		font-weight: 200;
		font-size: 2.5rem;
	}
	.network-diagram {
		display: flex;
		justify-content: center;
		align-items: center;
		margin: 2em 0;
	}
	.network-diagram .layer {
		display: flex;
		flex-direction: column;
		justify-content: space-around;
		height: 150px;
	}
	.network-diagram .node {
		border: 1px solid black;
		border-radius: 50%;
		width: 50px;
		height: 50px;
		display: flex;
		justify-content: center;
		align-items: center;
		position: relative;
		background-color: white;
	}
	.network-diagram .node .value {
		position: absolute;
		bottom: -25px;
		font-size: 0.9em;
		color: #333;
	}
	.network-diagram .neuron {
		width: 80px;
		height: 80px;
	}
	.network-diagram .bias {
		position: absolute;
		bottom: -20px;
		font-size: 0.8em;
	}
	.network-diagram .arrows {
		width: 100px;
		height: 150px;
	}
	.network-diagram svg {
		width: 100%;
		height: 100%;
	}
	.controls {
		display: grid;
		grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
		gap: 1em;
		margin-bottom: 1em;
	}
	.controls div {
		display: flex;
		flex-direction: column;
	}
	.controls button {
		padding: 0.5em 1em;
		font-size: 1em;
		background-color: #007acc;
		color: white;
		border: none;
		border-radius: 5px;
		cursor: pointer;
		margin-top: 0.5em;
	}
	.controls button:hover {
		background-color: #005999;
	}
	.gradient-descent {
		margin: 2em 0;
		border: 1px solid #007acc;
		border-radius: 5px;
	}
	.gradient-descent summary {
		padding: 1em;
		font-weight: bold;
		cursor: pointer;
		background-color: #e8f4f8;
		color: #005999;
	}
	.gradient-descent-content {
		padding: 1em;
		border-top: 1px solid #007acc;
		background-color: #f8fcff;
	}
	.gd-controls {
		display: flex;
		flex-direction: column;
		gap: 1em;
	}
	.gd-controls > div {
		display: flex;
		flex-direction: column;
	}
	.gd-buttons {
		display: grid !important;
		grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
		gap: 0.5em;
		flex-direction: row !important;
	}
	.gd-buttons button {
		padding: 0.5em 1em;
		font-size: 1em;
		background-color: #007acc;
		color: white;
		border: none;
		border-radius: 5px;
		cursor: pointer;
	}
	.gd-buttons button:hover {
		background-color: #005999;
	}
	.gd-buttons button.stop {
		background-color: #dc3545;
	}
	.gd-buttons button.stop:hover {
		background-color: #c82333;
	}
	.info {
		margin-bottom: 1em;
		background: #f4f4f4;
		padding: 1em;
		border-radius: 5px;
	}
	.info .formula {
		font-size: 1.1em;
		font-style: italic;
		text-align: center;
		margin-bottom: 1em;
	}
	.info-grid {
		display: grid;
		grid-template-columns: 1fr 1fr;
		gap: 1em;
		text-align: center;
		margin-bottom: 1em;
	}
	.formula-grid {
		display: grid;
		grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
		gap: 1em;
		text-align: center;
	}
	.formula-grid p {
		margin: 0.2em 0;
	}
	.formula-grid .value {
		font-weight: bold;
		font-size: 1.2em;
	}
	.gradient-step-info {
		margin-top: 1em;
		padding: 1em;
		background: #fff;
		border-radius: 5px;
		border: 1px solid #007acc;
	}
	.gradient-step-info h4 {
		margin-top: 0;
		color: #005999;
	}
	.step-grid {
		display: grid;
		grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
		gap: 1em;
	}
	.step-value {
		font-family: monospace;
		background: #fff;
		padding: 0.5em;
		border-radius: 3px;
		border: 1px solid #ddd;
	}
	.next-value {
		font-weight: bold;
		color: #007acc;
	}
	.derivations {
		margin: 2em 0;
		border: 1px solid #ccc;
		border-radius: 5px;
	}
	.derivations summary {
		padding: 1em;
		font-weight: bold;
		cursor: pointer;
		background-color: #f9f9f9;
	}
	.derivation-content {
		padding: 1em;
		border-top: 1px solid #ccc;
	}
	.derivation-content h4 {
		margin-top: 0;
	}
	.derivation-content .formula {
		background-color: #eee;
		padding: 0.5em;
		border-radius: 3px;
		font-family: monospace;
	}
	.chart-grid {
		display: grid;
		grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
		gap: 1em;
	}
	.chart-container {
		position: relative;
		height: 40vh;
		width: 100%;
	}
</style>

