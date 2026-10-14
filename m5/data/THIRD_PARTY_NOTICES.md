# Third-party material in `m5/`

This repository is public. This file lists every third-party text embedded in `m5/`, where it came
from, and the notice its licence requires; what is deliberately **not** embedded and is read from
the operator's own copy instead; and the sources that have no licence. Text written for M5 is not
listed: it is marked `WRITTEN FOR M5` in the source. Each embedded text is cited in the source next
to where it is used (file, commit, line).

## 1. Licensed sources whose text is embedded

### 1.1 Pearson-Vogel et al., *latent-introspection-code* (ACS Research)

- Source: https://github.com/acsresearch/latent-introspection-code at commit
  5007f75766b6f8c650cab438af67f970a262ceca; paper arXiv 2602.20031.
- Taken, verbatim, in `m5/data/detection.py`:
  - `PRIMER`: the mechanism-informative primer, `INFO_INTROSPECT` (`main.py`:299-303).
  - Control questions `pearson_yes_1`-`8` (`CONTROL_QUESTIONS_YES`, `main.py`:620-629) and
    `pearson_varied_1`-`6` (`CONTROL_QUESTIONS_AMBIGUOUS`, `main.py`:632-639). The twenty
    `lindsey_*` questions are quoted from Lindsey (2026, appendix); the same strings are this
    repository's `CONTROL_QUESTIONS_NO` and `CONTROL_QUESTIONS_NO_CONFUSING` (`main.py`:591-617).
- Licence (the repository's `LICENSE`, verbatim):

```
Copyright (c) 2025 ACS Research

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE
OR OTHER DEALINGS IN THE SOFTWARE.
```

### 1.2 Hahami et al., *llama-introspection-new*

- Source: https://github.com/elyhahami18/llama-introspection-new at commit
  7d9280d79e4aa37599dc0fd89974be4bc7a54827; paper arXiv 2512.12411.
- Taken, verbatim, in `m5/data/detection.py`:
  - `SEGMENT_POOLS["hahami"]`: the 100 neutral sentences `LOCALIZATION_SENTENCES`
    (`code/utils/all_prompts.py`:389-490).
  - One sentence of `LOCALISATION_BRIEFING`, "I want to assess how well you can detect where I
    inject thoughts." (`code/utils/all_prompts.py`:545).
  - The control question `hahami_1plus1` (paper appendix; also in `all_prompts.py`).
- Licence (the repository's `LICENSE`, verbatim):

```
MIT License

Copyright (c) 2026 Ely Hahami

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### 1.3 Sridharan, *introspection* (a fork of Neev Parikh's replication)

- Source: https://github.com/agastyasridharan/introspection at commit
  501ee37d5f1cf44ca98dac2975be79ec7ac1fd2c.
- Taken, verbatim, in `m5/data/detection.py`: the factual control questions `sridharan_no_02`-`10`
  and `sridharan_yes_01`-`10` (`FACTUAL_NO_QUESTIONS` and `FACTUAL_YES_QUESTIONS`,
  `src/introspection/constants.py`:159-184; the first no-question is Lindsey's and is kept once).
  `m5/tests/test_fake_backend.py` also uses one of them, "Is the Earth flat?".
- Licence (the repository's `LICENSE`, verbatim; its copyright line names Neev Parikh):

```
MIT License

Copyright (c) 2025 Neev Parikh

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### 1.4 Chen et al., *Persona Vectors*

- Source: https://github.com/safety-research/persona_vectors at commit
  b8e0f044fe2410a6fad579f38324f03f13b4e917; paper arXiv 2507.21509. The repository has no `NOTICE`
  file.
- Taken, in `m5/judges/prompts.py`:
  - `_SCORE_LINES`: the trait judge's response instruction, verbatim (the `eval_prompt` of
    `data_generation/trait_data_eval/*.json`).
  - `TRAIT`: the trait judge prompt, **modified**: reordered so that the fixed instructions come
    first and the trait name, its description, the prompt and the response are supplied as
    labelled payload; the opening sentence reworded to name no trait; their sentence "Focus only
    on whether the model exhibits this trait." kept.
- `m5/judges/clients.py` reimplements their rule for reading a 0-100 score from log-probabilities
  (`judge.py` `_aggregate_0_100_score`); no code is copied.
- Licence: Apache License 2.0, "Copyright 2025 The Persona Vectors authors". The full text is in
  the appendix below, as the licence requires.

## 2. The vendored harness (Macar et al.)

`upstream/introspection_mechanisms/` is the authors' code (safety-research/introspection-mechanisms,
commit in `upstream/introspection_mechanisms/UPSTREAM_COMMIT`), vendored by an earlier operator
decision; no licence file is vendored with it. `m5/` embeds from it, verbatim and cited: the
detection conversation, the forced prefill and (as `MACAR_BASELINE_VERBATIM`) the 100 baseline
words in `m5/data/detection.py` and `m5/data/words.py`, and the judge prompts in
`m5/judges/macar.py`.

## 3. Short quotations from published papers and system cards (cited in the source)

- Claude Opus 4.6 system card §6.5.2: the *Testing vs. Conversation* and *Fiction vs. Real* pairs
  and the train pair (`m5/data/contrastive.py`; the train pair also in `m5/vectors.py`).
- Claude Fable 5 / Mythos 5 system card: five grader-awareness prompts and the train pair
  (`m5/data/contrastive.py`). The two prompts about exploiting an LLM judge are not taken.
- Lindsey (2026), *Emergent Introspective Awareness*: the twenty answer-is-no control questions,
  the fifty concepts and the baseline word list (`m5/data/detection.py`, `m5/data/words.py`).
- Macar et al. (2026): named concepts and their published detection rates (`m5/data/words.py`).

## 4. Not embedded: read at run time from the operator's own copy

- **Read et al., *glm-5-steering-logs*** (https://github.com/thjread-aisi/glm-5-steering-logs,
  commit 8e667d57fa79f09834264fda7ce4875bd397871a). The repository has no licence, and its README
  marks the data with canary strings against training corpora. Its ten unrelated controls and two
  purpose-of-evaluation pairs are recorded in `m5/data/contrastive.py` (`EXTERNAL_SETS`) by id,
  provenance, pair count and a sha256 over a canonical serialisation only, and are read from a
  local clone outside the repository (`load_external_sets`, config key `READ_DATA_PATH`), checked
  against those hashes. The repository's copies of the cards' own pairs are not used; the card
  quotations of section 3 are. No canary string, and no text from that repository's data, appears
  in `m5/` (`m5/tests/test_prompts_data.py` checks every file by hash).
- **Hua et al.'s sixteen evaluation-awareness pairs** (paper Table 12). Hua's repository
  (https://github.com/tim-hua-01/steering-eval-awareness-public, MIT, "Copyright (c) 2025 Tim
  Hua", commit c5fa5120c89434d48452c5c92032c66742f7ec6a) does not contain them: the authors
  removed the prompt files, and none of the 32 texts occurs in it. Their only machine-readable copy
  is Read et al.'s, so they are external in the same way (`hua_eval_awareness`,
  `hua_eval_awareness_best4`). Nothing from Hua's repository is embedded.
- **Panickssery et al., CAA** (https://github.com/nrimsky/CAA, MIT, "Copyright (c) 2024 Nina
  Rimsky"): `load_caa` reads the operator's CAA-format files. No CAA text is embedded; the twenty
  sycophancy items are written for M5 in CAA's format.
- **MMLU-Pro, GPQA Diamond, GSM8K**: read by `m5/data/capability.py` from the operator's files;
  nothing embedded.

## 5. Sources without a licence: nothing embedded

- **Nguyen et al., *evaluation-awareness-probing*** (https://github.com/Jordine/evaluation-awareness-probing,
  commit 5f901bf): no text from it occurs in `m5/`.
- **Black and Bloom, *llm-self-steering*** (https://github.com/UKGovernmentBEIS/llm-self-steering,
  commit fb65cbe): no text from it occurs in `m5/`. Its primer constant is the same text as
  Pearson-Vogel et al.'s; `m5`'s `PRIMER` is Pearson-Vogel et al.'s copy, byte for byte (theirs
  has curly quotes and an unspaced em dash; Black and Bloom's does not).
- **Read et al.**: section 4.

## Appendix: Apache License, Version 2.0 (from the Persona Vectors repository's `LICENSE`)

```
                                 Apache License
                           Version 2.0, January 2004
                        http://www.apache.org/licenses/

   TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

   1. Definitions.

      "License" shall mean the terms and conditions for use, reproduction,
      and distribution as defined by Sections 1 through 9 of this document.

      "Licensor" shall mean the copyright owner or entity authorized by
      the copyright owner that is granting the License.

      "Legal Entity" shall mean the union of the acting entity and all
      other entities that control, are controlled by, or are under common
      control with that entity. For the purposes of this definition,
      "control" means (i) the power, direct or indirect, to cause the
      direction or management of such entity, whether by contract or
      otherwise, or (ii) ownership of fifty percent (50%) or more of the
      outstanding shares, or (iii) beneficial ownership of such entity.

      "You" (or "Your") shall mean an individual or Legal Entity
      exercising permissions granted by this License.

      "Source" form shall mean the preferred form for making modifications,
      including but not limited to software source code, documentation
      source, and configuration files.

      "Object" form shall mean any form resulting from mechanical
      transformation or translation of a Source form, including but
      not limited to compiled object code, generated documentation,
      and conversions to other media types.

      "Work" shall mean the work of authorship, whether in Source or
      Object form, made available under the License, as indicated by a
      copyright notice that is included in or attached to the work
      (an example is provided in the Appendix below).

      "Derivative Works" shall mean any work, whether in Source or Object
      form, that is based on (or derived from) the Work and for which the
      editorial revisions, annotations, elaborations, or other modifications
      represent, as a whole, an original work of authorship. For the purposes
      of this License, Derivative Works shall not include works that remain
      separable from, or merely link (or bind by name) to the interfaces of,
      the Work and Derivative Works thereof.

      "Contribution" shall mean any work of authorship, including
      the original version of the Work and any modifications or additions
      to that Work or Derivative Works thereof, that is intentionally
      submitted to Licensor for inclusion in the Work by the copyright owner
      or by an individual or Legal Entity authorized to submit on behalf of
      the copyright owner. For the purposes of this definition, "submitted"
      means any form of electronic, verbal, or written communication sent
      to the Licensor or its representatives, including but not limited to
      communication on electronic mailing lists, source code control systems,
      and issue tracking systems that are managed by, or on behalf of, the
      Licensor for the purpose of discussing and improving the Work, but
      excluding communication that is conspicuously marked or otherwise
      designated in writing by the copyright owner as "Not a Contribution."

      "Contributor" shall mean Licensor and any individual or Legal Entity
      on behalf of whom a Contribution has been received by Licensor and
      subsequently incorporated within the Work.

   2. Grant of Copyright License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      copyright license to reproduce, prepare Derivative Works of,
      publicly display, publicly perform, sublicense, and distribute the
      Work and such Derivative Works in Source or Object form.

   3. Grant of Patent License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      (except as stated in this section) patent license to make, have made,
      use, offer to sell, sell, import, and otherwise transfer the Work,
      where such license applies only to those patent claims licensable
      by such Contributor that are necessarily infringed by their
      Contribution(s) alone or by combination of their Contribution(s)
      with the Work to which such Contribution(s) was submitted. If You
      institute patent litigation against any entity (including a
      cross-claim or counterclaim in a lawsuit) alleging that the Work
      or a Contribution incorporated within the Work constitutes direct
      or contributory patent infringement, then any patent licenses
      granted to You under this License for that Work shall terminate
      as of the date such litigation is filed.

   4. Redistribution. You may reproduce and distribute copies of the
      Work or Derivative Works thereof in any medium, with or without
      modifications, and in Source or Object form, provided that You
      meet the following conditions:

      (a) You must give any other recipients of the Work or
          Derivative Works a copy of this License; and

      (b) You must cause any modified files to carry prominent notices
          stating that You changed the files; and

      (c) You must retain, in the Source form of any Derivative Works
          that You distribute, all copyright, patent, trademark, and
          attribution notices from the Source form of the Work,
          excluding those notices that do not pertain to any part of
          the Derivative Works; and

      (d) If the Work includes a "NOTICE" text file as part of its
          distribution, then any Derivative Works that You distribute must
          include a readable copy of the attribution notices contained
          within such NOTICE file, excluding those notices that do not
          pertain to any part of the Derivative Works, in at least one
          of the following places: within a NOTICE text file distributed
          as part of the Derivative Works; within the Source form or
          documentation, if provided along with the Derivative Works; or,
          within a display generated by the Derivative Works, if and
          wherever such third-party notices normally appear. The contents
          of the NOTICE file are for informational purposes only and
          do not modify the License. You may add Your own attribution
          notices within Derivative Works that You distribute, alongside
          or as an addendum to the NOTICE text from the Work, provided
          that such additional attribution notices cannot be construed
          as modifying the License.

      You may add Your own copyright statement to Your modifications and
      may provide additional or different license terms and conditions
      for use, reproduction, or distribution of Your modifications, or
      for any such Derivative Works as a whole, provided Your use,
      reproduction, and distribution of the Work otherwise complies with
      the conditions stated in this License.

   5. Submission of Contributions. Unless You explicitly state otherwise,
      any Contribution intentionally submitted for inclusion in the Work
      by You to the Licensor shall be under the terms and conditions of
      this License, without any additional terms or conditions.
      Notwithstanding the above, nothing herein shall supersede or modify
      the terms of any separate license agreement you may have executed
      with Licensor regarding such Contributions.

   6. Trademarks. This License does not grant permission to use the trade
      names, trademarks, service marks, or product names of the Licensor,
      except as required for reasonable and customary use in describing the
      origin of the Work and reproducing the content of the NOTICE file.

   7. Disclaimer of Warranty. Unless required by applicable law or
      agreed to in writing, Licensor provides the Work (and each
      Contributor provides its Contributions) on an "AS IS" BASIS,
      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
      implied, including, without limitation, any warranties or conditions
      of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A
      PARTICULAR PURPOSE. You are solely responsible for determining the
      appropriateness of using or redistributing the Work and assume any
      risks associated with Your exercise of permissions under this License.

   8. Limitation of Liability. In no event and under no legal theory,
      whether in tort (including negligence), contract, or otherwise,
      unless required by applicable law (such as deliberate and grossly
      negligent acts) or agreed to in writing, shall any Contributor be
      liable to You for damages, including any direct, indirect, special,
      incidental, or consequential damages of any character arising as a
      result of this License or out of the use or inability to use the
      Work (including but not limited to damages for loss of goodwill,
      work stoppage, computer failure or malfunction, or any and all
      other commercial damages or losses), even if such Contributor
      has been advised of the possibility of such damages.

   9. Accepting Warranty or Additional Liability. While redistributing
      the Work or Derivative Works thereof, You may choose to offer,
      and charge a fee for, acceptance of support, warranty, indemnity,
      or other liability obligations and/or rights consistent with this
      License. However, in accepting such obligations, You may act only
      on Your own behalf and on Your sole responsibility, not on behalf
      of any other Contributor, and only if You agree to indemnify,
      defend, and hold each Contributor harmless for any liability
      incurred by, or claims asserted against, such Contributor by reason
      of your accepting any such warranty or additional liability.

   END OF TERMS AND CONDITIONS

   APPENDIX: How to apply the Apache License to your work.

      To apply the Apache License to your work, attach the following
      boilerplate notice, with the fields enclosed by brackets "[]"
      replaced with your own identifying information. (Don't include
      the brackets!)  The text should be enclosed in the appropriate
      comment syntax for the file format. We also recommend that a
      file or class name and description of purpose be included on the
      same "printed page" as the copyright notice for easier
      identification within third-party archives.

   Copyright 2025 The Persona Vectors authors

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
```
