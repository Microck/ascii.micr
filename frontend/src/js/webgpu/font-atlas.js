export class FontAtlas {
    constructor() {
        this.canvas = document.createElement('canvas');
        this.ctx = this.canvas.getContext('2d', { alpha: true });
        this.charWidth = 12;
        this.charHeight = 24;
        this.cols = 16;
        this.rows = 16;
        this.canvas.width = this.cols * this.charWidth;
        this.canvas.height = this.rows * this.charHeight;
        this.chars = []; 
    }

    async generate(font = 'monospace') {
        const chars = [];
        for (let i = 0; i < 256; i++) {
            chars.push(String.fromCharCode(i));
        }
        this.chars = chars;

        this.ctx.fillStyle = '#000000';
        this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
        
        this.ctx.font = `${this.charHeight - 4}px "${font}"`;
        this.ctx.textBaseline = 'top';
        this.ctx.fillStyle = '#FFFFFF';
        this.ctx.textAlign = 'center';

        for (let i = 0; i < 256; i++) {
            const char = chars[i];
            const col = i % this.cols;
            const row = Math.floor(i / this.cols);
            
            const x = col * this.charWidth;
            const y = row * this.charHeight;

            this.ctx.fillText(char, x + this.charWidth/2, y + 2);
        }

        return createImageBitmap(this.canvas);
    }

    getCharIndex(char) {
        return this.chars.indexOf(char);
    }
}
