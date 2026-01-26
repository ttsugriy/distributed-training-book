/**
 * Interactive Algorithm Stepper
 * Allows step-by-step visualization of algorithms like Ring AllReduce
 */

document$.subscribe(() => {
  // Initialize all steppers on page load
  document.querySelectorAll('.algorithm-stepper').forEach(initStepper);
});

function initStepper(container) {
  const steps = container.querySelectorAll('.step');
  const prevBtn = container.querySelector('.stepper-prev');
  const nextBtn = container.querySelector('.stepper-next');
  const stepIndicator = container.querySelector('.step-indicator');
  const playBtn = container.querySelector('.stepper-play');

  let currentStep = 0;
  let isPlaying = false;
  let playInterval = null;

  function updateDisplay() {
    steps.forEach((step, i) => {
      step.classList.toggle('active', i === currentStep);
    });

    if (stepIndicator) {
      stepIndicator.textContent = `Step ${currentStep + 1} of ${steps.length}`;
    }

    if (prevBtn) prevBtn.disabled = currentStep === 0;
    if (nextBtn) nextBtn.disabled = currentStep === steps.length - 1;

    if (playBtn) {
      playBtn.textContent = isPlaying ? '⏸ Pause' : '▶ Play';
    }
  }

  function goToStep(step) {
    currentStep = Math.max(0, Math.min(steps.length - 1, step));
    updateDisplay();
  }

  function next() {
    if (currentStep < steps.length - 1) {
      goToStep(currentStep + 1);
    } else if (isPlaying) {
      goToStep(0); // Loop back
    }
  }

  function prev() {
    goToStep(currentStep - 1);
  }

  function togglePlay() {
    isPlaying = !isPlaying;
    if (isPlaying) {
      playInterval = setInterval(next, 1500);
    } else {
      clearInterval(playInterval);
    }
    updateDisplay();
  }

  if (prevBtn) prevBtn.addEventListener('click', prev);
  if (nextBtn) nextBtn.addEventListener('click', next);
  if (playBtn) playBtn.addEventListener('click', togglePlay);

  // Keyboard navigation
  container.addEventListener('keydown', (e) => {
    if (e.key === 'ArrowRight') next();
    if (e.key === 'ArrowLeft') prev();
    if (e.key === ' ') { e.preventDefault(); togglePlay(); }
  });

  // Make container focusable for keyboard
  container.setAttribute('tabindex', '0');

  updateDisplay();
}

/**
 * Ring AllReduce Animation
 * Animates data flowing around a ring topology
 */
function initRingAnimation(container) {
  const svg = container.querySelector('svg');
  if (!svg) return;

  const nodes = svg.querySelectorAll('.ring-node');
  const dataPackets = svg.querySelectorAll('.data-packet');
  const phaseLabel = container.querySelector('.phase-label');

  let currentPhase = 0;
  const phases = ['Initial', 'ReduceScatter Step 1', 'ReduceScatter Step 2',
                  'ReduceScatter Step 3', 'AllGather Step 1', 'AllGather Step 2',
                  'AllGather Step 3', 'Complete'];

  function animate() {
    currentPhase = (currentPhase + 1) % phases.length;
    if (phaseLabel) {
      phaseLabel.textContent = phases[currentPhase];
    }

    // Trigger CSS animations
    dataPackets.forEach((packet, i) => {
      packet.style.setProperty('--phase', currentPhase);
    });
  }

  setInterval(animate, 2000);
}

// Auto-initialize ring animations
document$.subscribe(() => {
  document.querySelectorAll('.ring-animation').forEach(initRingAnimation);
});
