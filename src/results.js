/**
 * Results panel — updates the rank display, stats, confidence bar, and chart.
 */

import Chart from 'chart.js/auto';

let chartInstance = null;
let previousRank = null;

/** Animate a number counting up/down */
function animateNumber(element, targetValue, prefix = '#', duration = 600) {
  const startValue = previousRank || targetValue;
  const startTime = performance.now();

  function tick(now) {
    const elapsed = now - startTime;
    const progress = Math.min(elapsed / duration, 1);
    // Ease out cubic
    const eased = 1 - Math.pow(1 - progress, 3);
    const current = Math.round(startValue + (targetValue - startValue) * eased);
    element.textContent = `${prefix}${current}`;

    if (progress < 1) {
      requestAnimationFrame(tick);
    }
  }

  requestAnimationFrame(tick);
  previousRank = targetValue;
}

/** Update all result displays */
export function updateResults(results) {
  if (!results) return;

  const { medianRank, range90, scenarioScore, rankDistribution } = results;

  // 1. Rank display with animation
  const rankEl = document.getElementById('rank-display');
  if (rankEl) {
    animateNumber(rankEl, medianRank);
  }

  // Subtitle
  const subtitle = document.getElementById('rank-subtitle');
  if (subtitle) {
    subtitle.textContent = `Median from 5,000 simulations`;
  }

  // 2. Scenario Score
  const scoreEl = document.getElementById('scenario-score');
  if (scoreEl) {
    scoreEl.textContent = scenarioScore.toFixed(1);
  }

  // 3. CI Range
  const ciEl = document.getElementById('ci-range');
  if (ciEl) {
    ciEl.textContent = `${range90[0]}–${range90[1]}`;
  }

  // 4. Chart
  updateChart(rankDistribution, medianRank, range90);
}

/** Update or create the rank distribution chart */
function updateChart(distribution, medianRank, range90) {
  const canvas = document.getElementById('rankChart');
  if (!canvas) return;

  // Prepare data
  const ranks = Object.keys(distribution).map(Number).sort((a, b) => a - b);
  const probabilities = ranks.map(r => distribution[r] * 100);

  // Color bars: highlight median and CI range
  const colors = ranks.map(r => {
    if (r === medianRank) return 'rgba(99, 102, 241, 0.9)'; // indigo for median
    if (r >= range90[0] && r <= range90[1]) return 'rgba(6, 182, 212, 0.5)'; // cyan for CI
    return 'rgba(255, 255, 255, 0.1)'; // dim for outside
  });

  const borderColors = ranks.map(r => {
    if (r === medianRank) return 'rgba(99, 102, 241, 1)';
    if (r >= range90[0] && r <= range90[1]) return 'rgba(6, 182, 212, 0.8)';
    return 'rgba(255, 255, 255, 0.15)';
  });

  if (chartInstance) {
    // Update existing chart
    chartInstance.data.labels = ranks;
    chartInstance.data.datasets[0].data = probabilities;
    chartInstance.data.datasets[0].backgroundColor = colors;
    chartInstance.data.datasets[0].borderColor = borderColors;
    chartInstance.update('none');
  } else {
    // Create new chart
    chartInstance = new Chart(canvas, {
      type: 'bar',
      data: {
        labels: ranks,
        datasets: [{
          data: probabilities,
          backgroundColor: colors,
          borderColor: borderColors,
          borderWidth: 1,
          borderRadius: 2,
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          tooltip: {
            backgroundColor: 'rgba(15, 23, 42, 0.95)',
            borderColor: 'rgba(99, 102, 241, 0.3)',
            borderWidth: 1,
            titleColor: '#e2e8f0',
            bodyColor: '#94a3b8',
            cornerRadius: 8,
            padding: 12,
            callbacks: {
              title: (items) => `Rank #${items[0].label}`,
              label: (item) => `Probability: ${item.raw.toFixed(1)}%`,
            }
          }
        },
        scales: {
          x: {
            title: {
              display: true,
              text: 'Predicted Rank',
              color: '#64748b',
              font: { size: 11 }
            },
            ticks: {
              color: '#475569',
              font: { size: 10 },
              maxTicksLimit: 15,
            },
            grid: { display: false },
            border: { color: 'rgba(255,255,255,0.05)' },
          },
          y: {
            title: {
              display: true,
              text: 'Probability (%)',
              color: '#64748b',
              font: { size: 11 }
            },
            ticks: {
              color: '#475569',
              font: { size: 10 },
            },
            grid: {
              color: 'rgba(255,255,255,0.03)',
            },
            border: { display: false },
          }
        },
        animation: {
          duration: 300,
        },
      }
    });
  }
}

/** Show initial current GWU info */
export function showCurrentInfo(schoolName, rank, score) {
  const nameEl = document.getElementById('current-school-name');
  if (nameEl) nameEl.textContent = schoolName;

  const rankEl = document.getElementById('current-rank');
  if (rankEl) rankEl.textContent = `#${rank}`;

  const scoreEl = document.getElementById('current-score-label');
  if (scoreEl) scoreEl.textContent = `Score: ${score}`;

  const schoolLabel = document.getElementById('school-name');
  if (schoolLabel) {
    // Extract short name (e.g., "George Washington University" → "GWU's")
    const shortName = schoolName.includes('George Washington') ? "GWU's" : `${schoolName}'s`;
    schoolLabel.textContent = shortName;
  }
}
