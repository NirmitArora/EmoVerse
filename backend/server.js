const express = require("express");
const multer = require("multer");
const cors = require("cors");
const { exec } = require("child_process");
const path = require("path");
const fs = require("fs");

const app = express();
const PORT = 3000;

// Ensure the uploads directory exists
const uploadDir = path.join(__dirname, "uploads");
if (!fs.existsSync(uploadDir)) {
  fs.mkdirSync(uploadDir, { recursive: true });
}

app.use(cors());
app.use(express.static(path.join(__dirname, "..", "frontend")));

app.get("/", (req, res) => {
  res.sendFile(path.join(__dirname, "..", "frontend", "index.html"));
});
app.use(express.json({ limit: "10mb" }));

// Multer config: saves image with a unique timestamped name
const storage = multer.diskStorage({
  destination: function (req, file, cb) {
    cb(null, uploadDir);
  },
  filename: function (req, file, cb) {
    const uniqueName = `input_${Date.now()}.jpg`;
    cb(null, uniqueName);
  }
});
const upload = multer({ storage });

app.post("/detect", upload.single("image"), (req, res) => {
  const language = req.body.language || "english";
  const imagePath = req.file.path;
  const scriptPath = path.join(__dirname, "emotion.py");

  // Run the Python script with image path as an argument
  exec(`python "${scriptPath}" "${imagePath}"`, (err, stdout, stderr) => {
    // Delete the uploaded image after processing
    fs.unlink(imagePath, (unlinkErr) => {
      if (unlinkErr) console.warn("Could not delete uploaded image:", unlinkErr);
    });

    if (err) {
      console.error("Python error:", err);
      console.error("stderr:", stderr);
      return res.status(500).send("Emotion detection failed.");
    }

    const emotion = stdout.trim().toLowerCase();
    const query = `${language} ${emotion} songs`;
    const youtubeURL = `https://www.youtube.com/results?search_query=${encodeURIComponent(query)}`;

    res.json({ emotion, url: youtubeURL });
  });
});

app.listen(PORT, () => {
  console.log(`✅ Server running at http://localhost:${PORT}`);
});
