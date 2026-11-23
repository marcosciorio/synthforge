#pragma once

#include <juce_core/juce_core.h>

namespace SynthForgeClient
{
    constexpr int kPort = 51027;

    inline bool isServerReachable(int timeoutMs = 300)
    {
        juce::StreamingSocket socket;
        auto connected = socket.connect("127.0.0.1", kPort, timeoutMs);
        if (connected)
            socket.close();
        return connected;
    }

    struct Response
    {
        bool ok = false;
        juce::var params;
        juce::var x1;
        juce::String matchedPreset;
        double similarity = 0.0;
        juce::String error;
    };

    inline Response sendRequest(juce::DynamicObject::Ptr requestObj, int timeoutOverrideMs = -1)
    {
        Response result;

        juce::StreamingSocket socket;
        if (! socket.connect("127.0.0.1", kPort, 2000))
        {
            result.error = "Cannot connect to the SynthForge server on 127.0.0.1:"
                          + juce::String(kPort) + " - is it running? (python3 synthforge_server.py)";
            return result;
        }

        auto requestLine = juce::JSON::toString(juce::var(requestObj.get()), true) + "\n";
        auto utf8 = requestLine.toRawUTF8();
        if (socket.write(utf8, (int) strlen(utf8)) < 0)
        {
            result.error = "Error while writing to the SynthForge server socket.";
            return result;
        }

        int overallTimeoutMs;
        if (timeoutOverrideMs > 0)
        {
            overallTimeoutMs = timeoutOverrideMs;
        }
        else
        {
            int steps = (int) requestObj->getProperty("steps");
            overallTimeoutMs = juce::jlimit(15000, 180000, 10000 + steps * 400);
        }
        auto deadline = juce::Time::getMillisecondCounter() + (juce::uint32) overallTimeoutMs;

        juce::MemoryBlock accumulated;
        char buffer[4096];
        while (juce::Time::getMillisecondCounter() < deadline)
        {
            auto ready = socket.waitUntilReady(true, 500);
            if (ready < 0)
            {
                result.error = "Connection to the SynthForge server was interrupted.";
                return result;
            }
            if (ready == 0)
                continue;

            auto bytesRead = socket.read(buffer, (int) sizeof(buffer), false);
            if (bytesRead < 0)
            {
                result.error = "Error while reading from the SynthForge server.";
                return result;
            }
            if (bytesRead == 0)
                break;

            accumulated.append(buffer, (size_t) bytesRead);
            if (memchr(accumulated.getData(), '\n', accumulated.getSize()) != nullptr)
                break;
        }

        auto responseText = accumulated.toString();
        if (! responseText.contains("\n"))
        {
            result.error = "Timed out waiting for the SynthForge server response.";
            return result;
        }

        auto parsed = juce::JSON::parse(responseText);
        auto* obj = parsed.getDynamicObject();
        if (obj == nullptr)
        {
            result.error = "Invalid response from the SynthForge server.";
            return result;
        }

        result.ok = (bool) obj->getProperty("ok");
        if (result.ok)
        {
            result.params = obj->getProperty("params");
            result.x1 = obj->getProperty("x1");
            result.matchedPreset = obj->getProperty("matched_preset").toString();
            result.similarity = (double) obj->getProperty("similarity");
        }
        else
        {
            result.error = obj->getProperty("error").toString();
        }
        return result;
    }

    inline Response generate(const juce::String& prompt, int steps, double cfg)
    {
        juce::DynamicObject::Ptr req = new juce::DynamicObject();
        req->setProperty("action", "generate");
        req->setProperty("prompt", prompt);
        req->setProperty("steps", steps);
        req->setProperty("cfg", cfg);
        return sendRequest(req);
    }

    inline Response retrieveAndEdit(const juce::String& prompt, const juce::String& audioPath,
                                     int steps, double cfg, double tStar, double tau,
                                     const juce::String& category)
    {
        juce::DynamicObject::Ptr req = new juce::DynamicObject();
        req->setProperty("action", "retrieve_and_edit");
        req->setProperty("prompt", prompt);
        req->setProperty("audio_path", audioPath);
        req->setProperty("steps", steps);
        req->setProperty("cfg", cfg);
        req->setProperty("t_star", tStar);
        req->setProperty("tau", tau);
        req->setProperty("category", category);
        return sendRequest(req);
    }

    inline Response memoryEdit(const juce::String& prompt, int steps, double cfg, double tStar,
                                const juce::var& previousX1)
    {
        juce::DynamicObject::Ptr req = new juce::DynamicObject();
        req->setProperty("action", "memory_edit");
        req->setProperty("prompt", prompt);
        req->setProperty("steps", steps);
        req->setProperty("cfg", cfg);
        req->setProperty("t_star", tStar);
        req->setProperty("previous_x1", previousX1);
        return sendRequest(req);
    }

    inline Response generateFromAudio(const juce::String& audioPath, int steps, double cfg)
    {
        juce::DynamicObject::Ptr req = new juce::DynamicObject();
        req->setProperty("action", "generate_from_audio");
        req->setProperty("audio_path", audioPath);
        req->setProperty("steps", steps);
        req->setProperty("cfg", cfg);
        return sendRequest(req);
    }

    inline void registerClient()
    {
        juce::DynamicObject::Ptr req = new juce::DynamicObject();
        req->setProperty("action", "register");
        sendRequest(req, 3000);
    }

    inline void unregisterClient()
    {
        juce::DynamicObject::Ptr req = new juce::DynamicObject();
        req->setProperty("action", "unregister");
        sendRequest(req, 1000);
    }
}
