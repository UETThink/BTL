// Hiệu ứng hạt sáng theo chuột
class ParticleSystem {
    constructor(canvasId) {
        this.canvas = document.getElementById(canvasId);
        if (!this.canvas) return;
        this.ctx = this.canvas.getContext('2d');
        this.particles = [];
        this.mouse = { x: -100, y: -100 };
        this.hue = 220;
        this.resize();
        this.bindEvents();
        this.animate();
    }

    resize() {
        this.canvas.width = window.innerWidth;
        this.canvas.height = window.innerHeight;
    }

    bindEvents() {
        window.addEventListener('resize', () => this.resize());
        window.addEventListener('mousemove', (e) => {
            this.mouse.x = e.clientX;
            this.mouse.y = e.clientY;
            for (let i = 0; i < 3; i++) this.addParticle(e.clientX, e.clientY);
        });
        window.addEventListener('click', (e) => {
            for (let i = 0; i < 20; i++) this.addParticle(e.clientX, e.clientY, true);
        });
    }

    addParticle(x, y, burst = false) {
        const angle = Math.random() * Math.PI * 2;
        const speed = burst ? Math.random() * 4 + 1 : Math.random() * 1.5 + 0.3;
        this.particles.push({
            x, y,
            vx: Math.cos(angle) * speed,
            vy: Math.sin(angle) * speed,
            life: 1,
            decay: burst ? 0.02 + Math.random() * 0.02 : 0.008 + Math.random() * 0.012,
            size: burst ? Math.random() * 5 + 2 : Math.random() * 3 + 1,
            hue: this.hue + Math.random() * 60 - 30
        });
        if (this.particles.length > 200) this.particles.splice(0, 5);
    }

    animate() {
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        this.hue = (this.hue + 0.1) % 360;

        for (let i = this.particles.length - 1; i >= 0; i--) {
            const p = this.particles[i];
            p.x += p.vx;
            p.y += p.vy;
            p.vy += 0.02;
            p.life -= p.decay;

            if (p.life <= 0) { this.particles.splice(i, 1); continue; }

            this.ctx.save();
            this.ctx.globalAlpha = p.life * 0.7;
            this.ctx.fillStyle = `hsla(${p.hue}, 80%, 65%, ${p.life})`;
            this.ctx.shadowColor = `hsla(${p.hue}, 90%, 60%, 0.5)`;
            this.ctx.shadowBlur = 15;
            this.ctx.beginPath();
            this.ctx.arc(p.x, p.y, p.size * p.life, 0, Math.PI * 2);
            this.ctx.fill();
            this.ctx.restore();
        }
        requestAnimationFrame(() => this.animate());
    }
}

document.addEventListener('DOMContentLoaded', () => {
    new ParticleSystem('particle-canvas');

    // Cursor glow
    const glow = document.getElementById('cursor-glow');
    if (glow) {
        document.addEventListener('mousemove', (e) => {
            glow.style.left = e.clientX + 'px';
            glow.style.top = e.clientY + 'px';
        });
    }
});
